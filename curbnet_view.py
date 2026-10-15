"""Run CurbNet inference on one point cloud and display every point in Open3D."""

import argparse

import numpy as np

from inference import CURB_CLASS_ID, CurbNetInference, load_point_cloud

DEFAULT_CHECKPOINT = "1015_epoch6_model_curb_seq0.2_0519_4c_improve510_e100.pt"

def main():
    parser = argparse.ArgumentParser(
        description="Run CurbNet inference and open a full-resolution Open3D view."
    )
    parser.add_argument("path", help="Input .e57 or .ply point-cloud file.")
    parser.add_argument("--checkpoint", help="Checkpoint path (overrides the config).", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", help="CurbNet YAML config path.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Inference points per chunk (default: CURBNET_INFERENCE_CHUNK_POINTS or 250000).",
    )
    args = parser.parse_args()

    try:
        import open3d as o3d
    except ImportError:
        parser.error(
            "Open3D is required for this viewer. Install it with "
            "'pip install open3d==0.18.0'."
        )

    model = CurbNetInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        chunk_size=args.chunk_size,
    )
    points = load_point_cloud(args.path)
    predictions = model.predict_points(points)

    curb_mask = predictions == CURB_CLASS_ID
    colors = np.full((len(points), 3), (0.62, 0.67, 0.72), dtype=np.float64)
    colors[curb_mask] = (0.0, 0.9, 0.3)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors)
    print(
        "Displaying {:,} points; {:,} predicted as curb.".format(
            len(points), int(np.count_nonzero(curb_mask))
        )
    )
    o3d.visualization.draw_geometries(
        [cloud],
        window_name="CurbNet predictions (green = curb)",
        point_show_normal=False,
    )


if __name__ == "__main__":
    main()
