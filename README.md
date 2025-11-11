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
Inspired by Segment Anything 2, which generalizes segmentation from images to videos, we propose SAM2MOT—a novel segmentation-driven paradigm for multi-object tracking that breaks away from the conventional detection-association framework. In contrast to previous approaches that treat segmentation as auxiliary information, SAM2MOT places it at the heart of the tracking process, systematically tackling challenges like false positives and occlusions. Its effectiveness has been thoroughly validated on major MOT benchmarks. Furthermore, SAM2MOT integrates pre-trained detector, pre-trained segmentor with tracking logic into a zero-shot MOT system that requires no fine-tuning. This significantly reduces dependence on labeled data and paves the way for transitioning MOT research from task-specific solutions to general-purpose systems. Experiments on DanceTrack, UAVDT, and BDD100K show state-of-the-art results. Notably, SAM2MOT outperforms existing methods on DanceTrack by +2.1 HOTA and +4.5 IDF1, highlighting its effectiveness in MOT.



## Tracking performance
### Results on DanceTrack test set
| Detector       | HOTA | IDF1 | MOTA | AssA | DetA | TP     | FN    | FP    | IDSW |
|----------------|------|------|------|------|------|--------|-------|-------|------|
|co-dino-l       | 75.5 | 83.4 | 89.2 | 71.3 | 80.3 | 274582 | 14584 | 15653 | 854  |
|grouding-dino-l | 75.8 | 83.9 | 88.5 | 72.2 | 79.7 | 271472 | 17694 | 14650 | 879  |

### Results on UAVDT test set
| Detector       | Eval-IOU | MOTA | IDF1 | TP     | FN     | FP    | IDSW | MT  | ML  |
|----------------|----------|------|------|--------|--------|-------|------|-----|-----|
|co-dino-l       | 0.5      | 55.6 | 74.4 | 248402 | 92504  | 58610 | 141  | 742 | 161 |
|co-dino-l       | 0.4      | 66.1 | 79.3 | 266320 | 74586  | 40692 | 136  | 816 | 147 |
|grouding-dino-l | 0.5      | 51.0 | 71.7 | 236929 | 103977 | 62906 | 139  | 694 | 189 |
|grouding-dino-l | 0.4      | 60.9 | 76.6 | 253903 | 87003  | 45932 | 155  | 767 | 171 |


## Acknowledgment

SAM2MOT is built on top of [SAM 2](https://github.com/facebookresearch/sam2?tab=readme-ov-file) by Meta FAIR.

## Citation

Please consider citing our paper and the wonderful `SAM 2` if you found our work interesting and useful.
```
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}

@article{jiang2025sam2mot,
  title={SAM2MOT: A Novel Paradigm of Multi-Object Tracking by Segmentation},
  author={Jiang, Junjie and Wang, Zelin and Zhao, Manqi and Li, Yin and Jiang, DongSheng},
  journal={arXiv preprint arXiv:2504.04519},
  year={2025}
}
```
