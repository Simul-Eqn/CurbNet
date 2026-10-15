"""Small Flask UI and REST API for CurbNet point-cloud inference."""

import logging
import os
import re
import tempfile
import threading
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from inference import CURB_CLASS_ID, CurbNetInference, MAX_UPLOAD_BYTES

DEFAULT_CHECKPOINT = "1015_epoch6_model_curb_seq0.2_0519_4c_improve510_e100.pt"

RESULTS_DIR = Path(os.environ.get(
    "CURBNET_RESULTS_DIR",
    str(Path(tempfile.gettempdir()) / "curbnet-results"),
)).resolve()
UPLOADS_DIR = Path(os.environ.get(
    "CURBNET_UPLOADS_DIR",
    str(Path(tempfile.gettempdir()) / "curbnet-uploads"),
)).resolve()
VISUALIZATION_POINT_LIMIT = 60000


def _spatially_stratified_indices(points, candidates, limit, rng):
    if len(candidates) <= limit:
        return candidates
    if limit <= 0:
        return np.empty(0, dtype=np.int64)

    xyz = points[:, :3]
    lower = np.min(xyz, axis=0)
    span = np.ptp(xyz, axis=0)
    span[span == 0] = 1
    bins_per_axis = max(1, int(round(limit ** (1 / 3))))
    candidate_xyz = xyz[candidates]
    cell_coordinates = np.floor(
        (candidate_xyz - lower) / span * bins_per_axis
    ).astype(np.int64)
    np.clip(cell_coordinates, 0, bins_per_axis - 1, out=cell_coordinates)
    cell_keys = np.ravel_multi_index(cell_coordinates.T, (bins_per_axis,) * 3)

    sorted_positions = np.argsort(cell_keys, kind="stable")
    sorted_keys = cell_keys[sorted_positions]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(sorted_positions)])

    if len(starts) >= limit:
        selected_groups = np.linspace(0, len(starts) - 1, limit, dtype=np.int64)
        selected_positions = (
            starts[selected_groups]
            + rng.integers(0, counts[selected_groups])
        )
        return candidates[sorted_positions[selected_positions]]

    representative_positions = starts + rng.integers(0, counts)
    sampled_mask = np.ones(len(candidates), dtype=bool)
    sampled_mask[sorted_positions[representative_positions]] = False
    remaining = limit - len(representative_positions)
    available = candidates[sampled_mask]
    extra = rng.choice(available, size=remaining, replace=False)
    representatives = candidates[sorted_positions[representative_positions]]
    return np.concatenate((representatives, extra))


def _visualization_indices(points, predictions, limit):
    if limit <= 0 or len(predictions) <= limit:
        if limit > 0:
            return np.arange(len(predictions), dtype=np.int64)
        return np.empty(0, dtype=np.int64)

    curb_indices = np.flatnonzero(predictions == CURB_CLASS_ID)
    other_indices = np.flatnonzero(predictions != CURB_CLASS_ID)
    rng = np.random.default_rng(0)
    curb_limit = min(len(curb_indices), limit // 2)
    other_limit = min(len(other_indices), limit - curb_limit)
    remaining = limit - curb_limit - other_limit
    curb_limit += min(remaining, len(curb_indices) - curb_limit)

    sampled_curb = _spatially_stratified_indices(
        points, curb_indices, curb_limit, rng
    )
    sampled_other = _spatially_stratified_indices(
        points, other_indices, other_limit, rng
    )
    return np.concatenate((sampled_curb, sampled_other))


def create_app(inference_model=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    inference_model = inference_model or CurbNetInference(
        checkpoint_path=os.environ.get("CURBNET_CHECKPOINT", DEFAULT_CHECKPOINT),
        config_path=os.environ.get("CURBNET_CONFIG") or None,
    )
    inference_lock = threading.Lock()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ready"})

    @app.post("/api/predict")
    def predict():
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            return jsonify({"error": "Choose an .e57 or .ply file."}), 400
        filename = secure_filename(uploaded.filename)
        if Path(filename).suffix.lower() not in (".e57", ".ply"):
            return jsonify({"error": "Only .e57 and .ply files are supported."}), 415

        upload_path = UPLOADS_DIR / (uuid.uuid4().hex + Path(filename).suffix.lower())
        result_id = uuid.uuid4().hex
        result_path = RESULTS_DIR / (result_id + ".npy")
        try:
            uploaded.save(str(upload_path))
            with inference_lock:
                result = inference_model.predict_file(upload_path)
            predictions = result["predictions"]
            np.save(str(result_path), predictions, allow_pickle=False)

            sample_indices = _visualization_indices(
                result["points"], predictions, VISUALIZATION_POINT_LIMIT
            )
            visualization = {
                "xyz": result["points"][sample_indices].tolist(),
                "labels": predictions[sample_indices].tolist(),
            }
            curb_count = int(np.count_nonzero(result["curb_mask"]))
            return jsonify({
                "point_count": int(len(predictions)),
                "curb_count": curb_count,
                "curb_model_class": CURB_CLASS_ID,
                "curb_raw_label": 3,
                "visualization": visualization,
                "labels_url": "/api/results/{}".format(result_id),
            })
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            app.logger.exception("CurbNet could not process the uploaded cloud.")
            return jsonify({"error": str(exc)}), 500
        except Exception:
            app.logger.exception("Inference failed for uploaded point cloud.")
            return jsonify({"error": "Inference failed. Check the server log for details."}), 500
        finally:
            try:
                upload_path.unlink(missing_ok=True)
            except OSError:
                app.logger.exception("Could not remove temporary upload %s.", upload_path)

    @app.get("/api/results/<result_id>")
    def download_result(result_id):
        if not re.fullmatch(r"[0-9a-f]{32}", result_id):
            return jsonify({"error": "Invalid result ID."}), 400
        result_path = RESULTS_DIR / (result_id + ".npy")
        if not result_path.is_file():
            return jsonify({"error": "Result not found."}), 404
        return send_file(
            str(result_path),
            mimetype="application/octet-stream",
            as_attachment=True,
            download_name="curbnet_point_labels.npy",
        )

    @app.errorhandler(413)
    def upload_too_large(_error):
        return jsonify({"error": "Upload exceeds the configured 12 GiB limit."}), 413

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    flask_app = create_app()
    flask_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
