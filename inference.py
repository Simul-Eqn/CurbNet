"""Inference and point-cloud loading for the CurbNet 0.2 checkpoint."""

import os
from pathlib import Path

import numpy as np
import torch

from builder import model_builder
from config.config import load_config_data


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "semantickitti-curb_0.2.yaml"
CURB_CLASS_ID = 1
MAX_UPLOAD_BYTES = 12 * 1024 ** 3
DEFAULT_INFERENCE_CHUNK_POINTS = 250_000


def _point_array(xyz, intensity):
    """Return only XYZ and intensity; RGB/color fields are never model inputs."""
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError("Point cloud must contain x, y, and z coordinates.")

    xyz = np.asarray(xyz[:, :3], dtype=np.float32)
    if intensity is None:
        raise ValueError("Point cloud must provide LiDAR intensity/reflectance.")
    intensity = np.asarray(intensity, dtype=np.float32).reshape(-1)
    if len(intensity) != len(xyz):
        raise ValueError("Intensity count does not match the point count.")
    return np.column_stack((xyz, intensity))


def load_point_cloud(file_path):
    """Load E57 or PLY points as [x, y, z, intensity], deliberately excluding RGB."""
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".ply":
        try:
            from plyfile import PlyData
        except ImportError as exc:
            raise RuntimeError("PLY input requires the 'plyfile' package.") from exc

        vertex = PlyData.read(str(path))["vertex"].data
        fields = vertex.dtype.names or ()
        missing = [name for name in ("x", "y", "z") if name not in fields]
        if missing:
            raise ValueError("PLY vertex data is missing coordinate field(s): " + ", ".join(missing))
        # Do not read red/green/blue; only XYZ and optional reflectance/intensity are used.
        intensity_field = next(
            (name for name in ("intensity", "reflectance") if name in fields), None
        )
        if intensity_field is None:
            raise ValueError("PLY vertex data must provide intensity or reflectance.")
        intensity = vertex[intensity_field]
        return _point_array(
            np.column_stack((vertex["x"], vertex["y"], vertex["z"])),
            intensity,
        )

    if extension == ".e57":
        try:
            import pye57
        except ImportError as exc:
            raise RuntimeError("E57 input requires the 'pye57' package.") from exc

        e57 = pye57.E57(str(path))
        scans = []
        for scan_index in range(e57.scan_count):
            scan = e57.read_scan(
                scan_index,
                ignore_missing_fields=True,
                colors=False,
                intensity=True,
            )
            if not all(name in scan for name in ("cartesianX", "cartesianY", "cartesianZ")):
                raise ValueError("E57 scan is missing Cartesian XYZ coordinates.")
            if "intensity" not in scan:
                raise ValueError("Each E57 scan must provide LiDAR intensity.")
            coordinates = np.column_stack(
                (scan["cartesianX"], scan["cartesianY"], scan["cartesianZ"])
            )
            intensity = scan["intensity"]
            # Color loading is disabled; the checkpoint uses XYZ plus intensity only.
            scans.append(_point_array(coordinates, intensity))
        if not scans:
            raise ValueError("E57 file contains no scans.")
        return np.concatenate(scans, axis=0) if len(scans) > 1 else scans[0]

    raise ValueError("Unsupported point-cloud format. Upload a .e57 or .ply file.")


class CurbNetInference:
    """Load the configured CurbNet model and predict one complete point cloud."""

    def __init__(self, checkpoint_path=None, config_path=None, device=None, chunk_size=None):
        self.config_path = Path(config_path or DEFAULT_CONFIG).resolve()
        self.config = load_config_data(str(self.config_path))
        self.model_config = self.config["model_params"]
        self.dataset_config = self.config["dataset_params"]

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if chunk_size is None:
            chunk_size = int(os.environ.get(
                "CURBNET_INFERENCE_CHUNK_POINTS",
                DEFAULT_INFERENCE_CHUNK_POINTS,
            ))
        if not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
            raise ValueError("Inference chunk size must be a positive integer.")
        self.chunk_size = int(chunk_size)

        default_checkpoint = self.config["train_params"]["model_load_path"]
        self.checkpoint_path = Path(checkpoint_path or default_checkpoint)
        if not self.checkpoint_path.is_absolute():
            self.checkpoint_path = (PROJECT_ROOT / self.checkpoint_path).resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError("CurbNet checkpoint not found: {}".format(self.checkpoint_path))

        self.model = model_builder.build(self.model_config)
        checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device)
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.grid_size = np.asarray(self.model_config["output_shape"], dtype=np.int64)
        self.min_bound = np.asarray(self.dataset_config["min_volume_space"], dtype=np.float32)
        self.max_bound = np.asarray(self.dataset_config["max_volume_space"], dtype=np.float32)
        if self.model_config["fea_dim"] != 9:
            raise ValueError(
                "This checkpoint's point input must have 9 features "
                "(geometry plus LiDAR intensity; no RGB)."
            )

    def predict_points(self, points):
        """Return one predicted model class ID per input point."""
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 4 or len(points) == 0:
            raise ValueError("Expected a non-empty Nx4 point array: XYZ plus intensity.")
        if not np.isfinite(points).all():
            raise ValueError("Point coordinates and intensity must be finite.")

        xyz = points[:, :3]
        # Input column four is LiDAR intensity/reflectance, never RGB.
        intensity = points[:, 3]
        predictions = np.empty(len(points), dtype=np.uint8)
        intervals = (self.max_bound - self.min_bound) / (self.grid_size - 1)
        with torch.no_grad():
            for start in range(0, len(points), self.chunk_size):
                stop = min(start + self.chunk_size, len(points))
                xyz_chunk = xyz[start:stop]
                chunk_intensity = intensity[start:stop]

                rho = np.sqrt(xyz_chunk[:, 0] ** 2 + xyz_chunk[:, 1] ** 2)
                phi = np.arctan2(xyz_chunk[:, 1], xyz_chunk[:, 0])
                xyz_polar = np.column_stack((rho, phi, xyz_chunk[:, 2])).astype(np.float32)
                grid_indices = np.floor(
                    (np.clip(xyz_polar, self.min_bound, self.max_bound) - self.min_bound)
                    / intervals
                ).astype(np.int64)

                voxel_centers = (
                    (grid_indices.astype(np.float32) + 0.5) * intervals + self.min_bound
                )
                # Matches cylinder_dataset: centered polar XYZ, polar XYZ, Cartesian XY, intensity.
                point_features = np.column_stack(
                    (
                        xyz_polar - voxel_centers,
                        xyz_polar,
                        xyz_chunk[:, :2],
                        chunk_intensity,
                    )
                ).astype(np.float32)

                feature_tensor = torch.from_numpy(point_features).to(self.device)
                # torch indexing and the sparse model coordinate path require int64 indices.
                grid_tensor = torch.from_numpy(grid_indices).to(
                    device=self.device, dtype=torch.long
                )
                logits = self.model([feature_tensor], [grid_tensor], batch_size=1)
                voxel_predictions = torch.argmax(logits[0], dim=0)
                point_predictions = voxel_predictions[
                    grid_tensor[:, 0], grid_tensor[:, 1], grid_tensor[:, 2]
                ]
                predictions[start:stop] = point_predictions.cpu().numpy().astype(np.uint8)
                del feature_tensor, grid_tensor, logits, voxel_predictions, point_predictions
        return predictions

    def predict_file(self, file_path):
        """Load an E57/PLY and return its XYZ/intensity points and point predictions."""
        points = load_point_cloud(file_path)
        predictions = self.predict_points(points)
        return {
            "points": points[:, :3],
            "intensity": points[:, 3],
            "predictions": predictions,
            "curb_mask": predictions == CURB_CLASS_ID,
        }
