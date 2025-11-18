<div align="center">

# SAM2MOT: A Novel Paradigm of Multi-Object Tracking by Segmentation

Junjie Jiang, Zelin Wang, Manqi Zhao, Yin Li, Dongsheng Jiang

Intelligent Computing Algorithm Innovation Lab, Huawei Cloud

*[arXiv 2504.04519](https://arxiv.org/abs/2504.04519)*
</div>

<div align="center">

[![python](https://img.shields.io/badge/python-3.10%2B-green)]()
[![pytorch](https://img.shields.io/badge/pytorch-2.5.1%2B-orange)]()

</div>

This repository is the official implementation of SAM2MOT: A Novel Paradigm of Multi-Object Tracking by Segmentation

## News

- [ ] **Incoming**: We will release our code soon. Stay tuned.
- [ ] **Incoming**: We will publish an updated version of our paper on arxiv.
- [x] **2025/11/08**: Our paper is accepted by AAAI 2026!
- [x] **2025/04/06**: Release [paper](https://arxiv.org/abs/2504.04519)

https://github.com/user-attachments/assets/251346be-e664-44be-b89a-3f0970115dba

![compare results](assets/demo-image.png)

<div align="center">


</div>

## Abstract

Inspired by Segment Anything 2, which generalizes segmentation from images to videos, we propose SAM2MOT—a novel
segmentation-driven paradigm for multi-object tracking that breaks away from the conventional detection-association
framework. In contrast to previous approaches that treat segmentation as auxiliary information, SAM2MOT places it at the
heart of the tracking process, systematically tackling challenges like false positives and occlusions. Its effectiveness
has been thoroughly validated on major MOT benchmarks. Furthermore, SAM2MOT integrates pre-trained detector, pre-trained
segmentor with tracking logic into a zero-shot MOT system that requires no fine-tuning. This significantly reduces
dependence on labeled data and paves the way for transitioning MOT research from task-specific solutions to
general-purpose systems. Experiments on DanceTrack, UAVDT, and BDD100K show state-of-the-art results. Notably, SAM2MOT
outperforms existing methods on DanceTrack by +2.1 HOTA and +4.5 IDF1, highlighting its effectiveness in MOT.

## Installation

### 1.Create and activate environment

```bash
conda create -n sam2mot python=3.10 -y
conda activate sam2mot
```

### 2. Install PyTorch (choose a version suitable for your environment)

SAM2MOT requires torch >= 2.5.0.
Below is one valid option for CUDA 12.1:

```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

You may install any compatible PyTorch version depending on your CUDA setup.

### 3. install SAM2

```bash
cd SAM2
pip install -e .
pip install -e ".[notebooks]"
cd ..
```

### 4. Install MMDetection ecosystem

```bash
pip install -U openmim
mim install mmengine==0.10.5
mim install mmcv==2.1.0
mim install mmdet==3.3.0
```

### 5. Install additional dependencies

```bash
pip install -r requirements.txt
```

## Directory structure
```text
sam2mot/
├─ assets/
├─ checkpoints/
│  └─ co_dino_5scale_swin_large_16e_o365tococo.pth
│  └─ grounding_dino_swin-l_pretrain_all-56d69e78.pth
│  └─ sam2.1_hiera_large.pt
├─ config/
│  └─ co_dino.yml
│  └─ grounding_dino.yml
│  └─ dancetrack_test\
│     └─ co_dino_detections_thresholds.csv
│     └─ grounding_dino_detections_thresholds.csv
├─ data/
│  └─ demo/
│     └─ 00000001.jpg
│     └─ 00000002.jpg
│     └─ ...
├─ datasets/
│  └─ dancetrack/
│     └─ test
│        └─ dancetrack0003
│        └─ dancetrack0009
├─ results/
│  └─ demo/
│     └─ dancetrack
│     └─ demo
│        └─ co_dino
│           └─ detections.txt
│           └─ trajectories.txt
│           └─ visualization.mp4
│           └─ masks
│              └─ frame_00001_masks.npz
│              └─ frame_00002_masks.npz
│              └─ ...
│        └─ grounding_dino
├─ mmdetection/
├─ SAM2/
├─ tracker/
├─ run_dancetrack_test_inference.py
├─ run_demo.py
├─ sam2mot.py
├─ visualization.py
└─ requirements.txt
```

## Run Demo

### Using Co-DINO as detector
```bash
python run_demo.py \
  --device cuda:0 \
  --config config/co_dino.yml \
  --imgs_dir <path_to_images> \
  --det_conf_threshold <confidence_threshold> \
  --output_path <output_directory>
```

### Using GroundingDINO as detector
```bash
python run_demo.py \
  --device cuda:0 \
  --config config/grounding_dino.yml \
  --imgs_dir <path_to_images> \
  --det_conf_threshold <confidence_threshold> \
  --output_path <output_directory>
```

## DanceTrack Inference

### Using CoDINO as detector

```bash
python run_dancetrack_test_inference.py \
  --device cuda:0 \
  --config config/co_dino.yml \
  --dataset_dir datasets/dancetrack/test \
  --datasets_thresholds config/dancetrack_test/co_dino_detections_thresholds.csv \
  --output_dir results/dancetrack/test 
```

### Using GroundingDINO as detector

```bash
python run_dancetrack_test_inference.py \
  --device cuda:0 \
  --config config/grounding_dino.yml \
  --dataset_dir datasets/dancetrack/test \
  --datasets_thresholds config/dancetrack_test/grounding_dino_detections_thresholds.csv \
  --output_dir results/dancetrack/test 
```

### Using pre-generated detections

```bash
python run_dancetrack_test_inference.py \
  --device cuda:0 \
  --config config/co_dino.yml \
  --dataset_dir datasets/dancetrack/test \
  --datasets_detections results/dancetrack/test/co_dino/detections \
  --datasets_thresholds config/dancetrack_test/co_dino_detections_thresholds.csv \
  --output_dir results/dancetrack/test 
```

## Tracking performance

### Results on DanceTrack test set

| Detector        | HOTA | IDF1 | MOTA | AssA | DetA | TP     | FN    | FP    | IDSW |
|-----------------|------|------|------|------|------|--------|-------|-------|------|
| co-dino-l       | 75.5 | 83.4 | 89.2 | 71.3 | 80.3 | 274582 | 14584 | 15653 | 854  |
| grouding-dino-l | 75.8 | 83.9 | 88.5 | 72.2 | 79.7 | 271472 | 17694 | 14650 | 879  |

### Results on UAVDT test set

| Detector        | Eval-IOU | MOTA | IDF1 | TP     | FN     | FP    | IDSW | MT  | ML  |
|-----------------|----------|------|------|--------|--------|-------|------|-----|-----|
| co-dino-l       | 0.5      | 55.6 | 74.4 | 248402 | 92504  | 58610 | 141  | 742 | 161 |
| co-dino-l       | 0.4      | 66.1 | 79.3 | 266320 | 74586  | 40692 | 136  | 816 | 147 |
| grouding-dino-l | 0.5      | 51.0 | 71.7 | 236929 | 103977 | 62906 | 139  | 694 | 189 |
| grouding-dino-l | 0.4      | 60.9 | 76.6 | 253903 | 87003  | 45932 | 155  | 767 | 171 |

## Acknowledgment

SAM2MOT is built on top of excellent open-source projects that made this work possible:

- [SAM2](https://github.com/facebookresearch/sam2) for video segmentation
- [MMDetection](https://github.com/open-mmlab/mmdetection) and its ecosystem

We thank the authors and communities behind these projects for their outstanding contributions.

## Citation

Please consider citing our paper if you found our work interesting and useful.

```
@article{jiang2025sam2mot,
  title={SAM2MOT: A Novel Paradigm of Multi-Object Tracking by Segmentation},
  author={Jiang, Junjie and Wang, Zelin and Zhao, Manqi and Li, Yin and Jiang, DongSheng},
  journal={arXiv preprint arXiv:2504.04519},
  year={2025}
}
```
