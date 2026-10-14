"""Inference and point-cloud loading for the CurbNet 0.2 checkpoint."""

import os
from pathlib import Path

import numpy as np
import torch
import yaml

from builder import model_builder
from config.config import load_config_data


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "semantickitti-curb_0.2.yaml"
CURB_CLASS_ID = 1
MAX_UPLOAD_BYTES = 12 * 1024 ** 3


def _point_array(xyz, intensity=None):
    """Return only XYZ and intensity; RGB/color fields are never model inputs."""
    xyz = np.column_stack((xyz["cartesianX"], xyz["cartesianY"], xyz["cartesianZ"])) \
        if getattr(xyz, "dtype", None) is not None and xyz.dtype.names else np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError("Point cloud must contain x, y, and z coordinates.")

    xyz = np.asarray(xyz[:, :3], dtype=np.float32)
    if intensity is None:
        intensity = np.zeros(len(xyz), dtype=np.float32)
    else:
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
        intensity = vertex[intensity_field] if intensity_field else None
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
            scan = e57.read_scan(scan_index, ignore_missing_fields=True)
            if not all(name in scan for name in ("cartesianX", "cartesianY", "cartesianZ")):
                raise ValueError("E57 scan is missing Cartesian XYZ coordinates.")
            coordinates = np.column_stack(
                (scan["cartesianX"], scan["cartesianY"], scan["cartesianZ"])
            )
            intensity = scan.get("intensity")
            # Color fields, if present in the E57, are intentionally ignored.
            scans.append(_point_array(coordinates, intensity))
        if not scans:
            raise ValueError("E57 file contains no scans.")
        return np.concatenate(scans, axis=0) if len(scans) > 1 else scans[0]

    raise ValueError("Unsupported point-cloud format. Upload a .e57 or .ply file.")


class CurbNetInference:
    """Load the configured CurbNet model and predict one complete point cloud."""

    def __init__(self, checkpoint_path=None, config_path=None, device=None):
        self.config_path = Path(config_path or DEFAULT_CONFIG).resolve()
        self.config = load_config_data(str(self.config_path))
        self.model_config = self.config["model_params"]
        self.dataset_config = self.config["dataset_params"]

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

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
        if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
            raise ValueError("Expected a non-empty Nx3 or Nx4 point array.")
        if not np.isfinite(points[:, :3]).all():
            raise ValueError("Point coordinates contain NaN or infinite values.")

        xyz = points[:, :3]
        # Input column four is LiDAR intensity/reflectance, never RGB.
        intensity = points[:, 3] if points.shape[1] >= 4 else np.zeros(len(points), np.float32)

        rho = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])
        xyz_polar = np.column_stack((rho, phi, xyz[:, 2])).astype(np.float32)

        intervals = (self.max_bound - self.min_bound) / (self.grid_size - 1)
        grid_indices = np.floor(
            (np.clip(xyz_polar, self.min_bound, self.max_bound) - self.min_bound) / intervals
        ).astype(np.int32)

        voxel_centers = (grid_indices.astype(np.float32) + 0.5) * intervals + self.min_bound
        # Matches cylinder_dataset: centered polar XYZ, polar XYZ, Cartesian XY, intensity.
        point_features = np.column_stack(
            (xyz_polar - voxel_centers, xyz_polar, xyz[:, :2], intensity)
        ).astype(np.float32)

        feature_tensor = torch.from_numpy(point_features).to(self.device)
        grid_tensor = torch.from_numpy(grid_indices).to(self.device)
        with torch.no_grad():
            logits = self.model([feature_tensor], [grid_tensor], batch_size=1)
            voxel_predictions = torch.argmax(logits[0], dim=0)
            point_predictions = voxel_predictions[
                grid_tensor[:, 0], grid_tensor[:, 1], grid_tensor[:, 2]
            ]
        return point_predictions.cpu().numpy().astype(np.uint8)

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
