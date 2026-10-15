
## CurbNet: Curb Detection Framework Based on LiDAR Point Cloud Segmentation

[![arXiv](https://img.shields.io/badge/arXiv-2403.16794-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2403.16794)
[![PR's Welcome](https://img.shields.io/badge/PRs-welcome-red.svg?style=flat)](https://github.com/guoyangzhao/CurbNet/pulls)


---

## 📌 Overview

Curb detection is a critical task in autonomous driving and robotic perception, enabling accurate understanding of road boundaries and drivable areas.

However, it remains challenging due to:
- (a) Difficulty in extracting **height-aware geometric features**
- (b) **Non-uniform density distribution** of LiDAR point clouds
- (c) Severe **class imbalance** between curb and non-curb points

To address these challenges, we propose **CurbNet**, a LiDAR-based curb detection framework with:

- A newly constructed **3D-Curb dataset**
- A **Multi-Scale Channel Attention (MSCA)** module for feature fusion and height modeling
- A **loss group strategy** to handle class imbalance
- A **post-processing refinement module** for improved prediction quality



## 🧠 Framework

<img src="https://github.com/guoyangzhao/CurbNet/blob/main/images/framework527.png" width="65%">


## 📊 Detection Results

### SemanticKITTI Benchmark

<img src="https://github.com/guoyangzhao/CurbNet/blob/main/images/3Dcurb-no-occ2.png" width="55%">



## 🗂️ Dataset: 3D-Curb

We construct the **3D-Curb dataset** based on the large-scale **SemanticKITTI dataset**, with the following features:

- Add a **new curb category (label = 3)** with full 3D annotations
- Preserve original **28 semantic classes**
- Captured using **64-line LiDAR**
- Designed for **urban autonomous driving scenarios**

<img src="https://github.com/guoyangzhao/CurbNet/blob/main/images/Dataset_construct527.png" width="55%">



### 📥 Download

- **Baidu Netdisk**: [Download Link](https://pan.baidu.com/s/1YKtiCdgugzTxHD6dTpXNHQ) (Code: 1234)  
- **Google Drive**: [Download Link](https://drive.google.com/file/d/18rK1TE96SfWMi2GjK0RO1U4UPxumA1-E/view?usp=sharing)



### 📦 Data Format

The dataset follows the **SemanticKITTI format**:

```

/kitti/dataset/
└── sequences/
├── 00/
│   ├── labels/
│   │     ├ 000000.label
│   │     └ 000001.label
│   └── velodyne/
│         ├ 000000.bin
│         └ 000001.bin
├── 01/
├── 02/
...
└── 10/

```

- Point clouds: `.bin`
- Labels: `.label`



### ⚠️ Important Note

Since **curb is an additional category**, we assign:

```

label: 3 → "curb"

```

To visualize correctly using SemanticKITTI API, please modify:

```

config/semantic-kitti.yaml

```

Add:

```

3: "curb"

```



### 🔧 Visualization Tool

We recommend using the official SemanticKITTI API:

👉 https://github.com/PRBonn/semantic-kitti-api



## 📁 Additional Dataset: NRS

The original **NRS dataset** contains only 2D annotations.

We:
- Project 2D labels to **3D point clouds**
- Convert format to **SemanticKITTI style**
- Release processed version for public use

### 📥 Download

- **Baidu Netdisk**: [Download Link](https://pan.baidu.com/s/1U6b5c6TxfruNT572_vwLYA) (Code: 1234)  
- **Google Drive**: [Download Link](https://drive.google.com/file/d/1kAj1xEHnppwrg2zLp42rCsLL8439glhy/view?usp=sharing)


---


## ⚙️ Installation

### Requirements

- Python ≥ 3.7
- PyTorch ≥ 1.2
- yaml
- Cython
- [torch-scatter](https://github.com/rusty1s/pytorch_scatter)
- [spconv](https://github.com/traveller59/spconv) (tested with `spconv==1.2.1`, `cuda==10.2`)
- [nuScenes-devkit](https://github.com/nutonomy/nuscenes-devkit) *(optional)*

The active Compose image (`Dockerfile.notchina`) uses a CUDA 11.4 system toolkit
and SpConv/Cumm built for CUDA 11.4. The closest official PyTorch wheel line is
CUDA 11.3 (`torch==1.10.0+cu113`); torch-scatter is matched to that PyTorch
wheel. Thus this is not a from-source PyTorch CUDA 11.4 build. The Docker build
prints installed framework/library versions and the local CUDA compiler version
to make the mixed wheel/toolkit versions explicit.



## 🚀 Training

1. Modify config file:

```

config/semantickitti-curb_0.2.yaml

````

2. Start training:

```bash
sh train_0.2.sh
````



## 🎯 Inference Demo

### Run demo on a folder of LiDAR scans:

```bash
python demo_folder_focal.py \
    --demo-folder YOUR_FOLDER \
    --save-folder YOUR_SAVE_FOLDER
```

### With labels (optional):

```bash
python demo_folder_focal.py \
    --demo-folder YOUR_FOLDER \
    --save-folder YOUR_SAVE_FOLDER \
    --demo-label-folder YOUR_LABEL_FOLDER
```

---

## 🌐 Point-cloud web inference

The Flask app serves an upload page and a REST endpoint for one complete `.e57` or
`.ply` point cloud. The 3D view is interactive and displays at most 60,000 sampled
points; its sample is spatially stratified, with curb points given priority to
preserve curb coverage across the scene. All per-point model labels can be
downloaded as a NumPy `.npy` file.

Install the web-specific packages into the same environment as CurbNet:

```bash
pip install -r requirements-web.txt
```

Ensure the checkpoint referenced by `train_params.model_load_path` in
`config/semantickitti-curb_0.2.yaml` is present, then start the app from the
repository root:

```bash
python app.py
```

Set `CURBNET_CHECKPOINT` to override the configured checkpoint path, or
`CURBNET_CONFIG` to use another config. Open `http://localhost:5000`.

The REST API accepts multipart form data with a `file` field at
`POST /api/predict`. It returns point/curb counts, a downsampled visualization,
and a `labels_url`; fetch that URL to download the full `uint8` prediction array
in original point order. Curb predictions use model class `1`, which corresponds
to raw dataset label `3` in this config.

The readers pass only XYZ and LiDAR intensity/reflectance to the model.
PLY/E57 RGB or color attributes are ignored, and intensity is required.
Files must use compatible metric coordinates and orientation. Uploads are
limited to 12 GiB. Inference processes points in chunks of 250,000 by default;
set `CURBNET_INFERENCE_CHUNK_POINTS` to another positive integer. Use a smaller
value to reduce peak GPU use, or a larger value if the GPU has enough memory.
Chunking can change predictions slightly because each model call sees only that
chunk's occupied voxels and local context. The complete upload and result are
still held in system memory, so large files also require sufficient host RAM.
SpConv selects each layer's algorithm when the layer is constructed; inference
does not mutate convolution algorithms after checkpoint weights are loaded,
because algorithm selection affects the weight layout.
The Flask development server is intended for local experimentation, not direct
public deployment.

### Full-resolution Open3D CLI viewer

To run inference from a terminal and interactively display every point, install
Open3D in the same environment:

```bash
pip install open3d==0.18.0
python curbnet_view.py path/to/cloud.e57
```

PLY is also supported. Optional `--checkpoint`, `--config`, and `--chunk-size`
arguments override the checkpoint/configuration and per-inference chunk size.
The Open3D window displays every input point, with predicted curb points in
green; very large clouds may require substantial system and graphics memory.

The Dockerfiles are unchanged. Install the packages in `requirements-web.txt`
manually into the CurbNet environment before starting the app. With the
provided Compose setup, start the container, then run
`python3 /app/files/CurbNet-main/app.py` inside it and open
`http://localhost:5000`. Compose binds that port to localhost only.

---

## ⭐ Citation

If you find this project useful, please consider giving a ⭐ and citing our work:

```bibtex
@article{zhao2025curbnet,
  title={CurbNet: Curb detection framework based on LiDAR point cloud segmentation},
  author={Zhao, Guoyang and Ma, Fulong and Qi, Weiqing and Liu, Yuxuan and Liu, Ming and Ma, Jun},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2025},
  publisher={IEEE}
}
```



## 📬 Contact

If you have any questions or suggestions, feel free to open an issue or contact the authors.



## 🌟 Acknowledgements

This project is built upon:

* SemanticKITTI dataset
* Open-source LiDAR perception frameworks [Cylinder3D](https://github.com/xinge008/Cylinder3D)

We thank the community for their contributions.
