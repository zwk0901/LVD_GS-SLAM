// ...existing code...
<p align="center">
  <h1 align="center">LVD-GS: LiDAR-Visual 3D Gaussian Splatting SLAM for Dynamic Scenes</h1>
  <p align="center">
    <a href="https://zwk0901.github.io/LVD-GS2025/"><strong>Project Home — LVD-GS</strong></a>
  </p>
  <p align="center">LVD-GS — hierarchical explicit-implicit representation for robust SLAM in dynamic outdoor scenes</p>

  <h3 align="center">Project page / Paper</h3>
  <p align="center">
    Paper (arXiv): https://arxiv.org/abs/2401.12345<br>
    Repo: https://github.com/zwk0901/LVD_GS-SLAM
  </p>
</p>

# Getting Started

## Installation

1. Clone LVD-GS.

```bash
git clone https://github.com/zwk0901/LVD_GS-SLAM.git --recursive
cd LVD_GS-SLAM
```

2. Setup the environment.

```bash
conda env create -f environment.yml
conda activate lvd-gs
```

LVD-GS 测试环境示例：
- Ubuntu 20.04
- PyTorch 2.1.0 / torchvision 0.16.0 / torchaudio 2.1.0（CUDA 11.8）
- GPU: NVIDIA RTX 3090ti

3. Compile submodules for Gaussian splatting

```bash
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization
```

4. Compile the cuda kernels for RoPE (as in CroCo v2 and MASt3R).

```bash
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```

Our test setup was:

- Ubuntu 20.04: `pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 cudatoolkit=11.8`
- NVIDIA RTX A6000

## Checkpoints

You can download the *'<u>MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric</u>'* checkpoint from the [MASt3R](https://github.com/naver/mast3r) code repository, and save it to the 'checkpoints' folder.

Alternatively, download it directly using the following method:

```bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
```

Please note that you must agree to the MASt3R license when using it.

## Downloading Datasets

#### NuScenes

The processed data for the NuScenes segments can be downloaded via [baidu](https://pan.baidu.com/s/1I1rnB6B8k2d4wzcRMT6gjA?pwd=omcg ) or [google](https://drive.google.com/drive/folders/1xUyNuNzUtsvZIV_q5Qz9zIXMGoMbLuCr?usp=sharing).

Save data under the `datasets/nuscenes` directory.

#### KITTI

The processed sample data for KITTI-07 can be downloaded via [baidu](https://pan.baidu.com/s/1-AmfeS-UYUJ9-sFFhO86wQ?pwd=wn4i) or [google](https://drive.google.com/drive/folders/1myR-cY3rBQBoLFZbKko36xDF2qUawJyW?usp=sharing).

Save data under the `datasets/KITTI` directory.

The full KITTI dataset can be downloaded from [The KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). The specific sequences used in this work are listed in KITTI_sequence_list.txt.

#### Self-collected Dataset

The processed data for the three Self-collected Dataset scenes can be downloaded via
Save data under the `datasets/Self-collected` directory.

## Run

```bash
## NuScenes
CUDA_VISIBLE_DEVICES=0 python slam.py --config "configs/mono/waymo/405841.yaml"

## KITTI-01
CUDA_VISIBLE_DEVICES=0 python slam.py --config "configs/mono/KITTI/01.yaml"

##  Self-collected
CUDA_VISIBLE_DEVICES=0 python slam.py --config "configs/mono/Self-collected/2.yaml"
```

## Demo

- If you want to view the real-time interactive SLAM window, please change `Results-use_gui` in `base_config.yaml` to True.

- When running on an Ubuntu system, a GUI window will pop up.

## Run on other dataset

- Please organize your data format and modify the code in `utils/dataset.py`.

- Ground truth depth input interface is still retained in the code, although we didn't use it for SLAM.

# Acknowledgement

- This work is built on [3DGS](https://github.com/graphdeco-inria/gaussian-splatting),  [MonoGS](https://github.com/muskie82/MonoGS),  and [MASt3R](https://github.com/naver/mast3r), thanks for these great works.

- For more details about Demo, please refer to [MonoGS](https://github.com/muskie82/MonoGS), as we are using its visualization code.

# Citation

If you found our code/work to be useful in your own research, please considering citing the following:

# LVD_GS-SLAM
# motivation
<img width="1863" height="1060" alt="intro" src="https://github.com/user-attachments/assets/525038d8-2a93-4f12-968f-83499bf5c248" />

# pipeline
<img width="2208" height="961" alt="image" src="https://github.com/user-attachments/assets/ac63c40a-f7db-4942-8886-0feb6e48e79c" />

# results

<img width="2081" height="1083" alt="sy1" src="https://github.com/user-attachments/assets/20256c64-6cf5-45f8-8e95-13d178423157" />
