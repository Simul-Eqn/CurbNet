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


RESULTS_DIR = Path(os.environ.get(
    "CURBNET_RESULTS_DIR",
    str(Path(tempfile.gettempdir()) / "curbnet-results"),
)).resolve()
UPLOADS_DIR = Path(os.environ.get(
    "CURBNET_UPLOADS_DIR",
    str(Path(tempfile.gettempdir()) / "curbnet-uploads"),
)).resolve()
VISUALIZATION_POINT_LIMIT = 60000


def _visualization_indices(predictions, limit):
    curb_indices = np.flatnonzero(predictions == CURB_CLASS_ID)
    other_indices = np.flatnonzero(predictions != CURB_CLASS_ID)
    curb_limit = min(len(curb_indices), limit // 2)
    other_limit = limit - curb_limit

    if len(curb_indices) > curb_limit:
        curb_indices = curb_indices[np.linspace(0, len(curb_indices) - 1, curb_limit, dtype=np.int64)]
    if len(other_indices) > other_limit:
        other_indices = other_indices[
            np.linspace(0, len(other_indices) - 1, other_limit, dtype=np.int64)
        ]
    return np.concatenate((curb_indices, other_indices))


def create_app(inference_model=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    inference_model = inference_model or CurbNetInference(
        checkpoint_path=os.environ.get("CURBNET_CHECKPOINT") or None,
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

            sample_indices = _visualization_indices(predictions, VISUALIZATION_POINT_LIMIT)
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
