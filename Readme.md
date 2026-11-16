<div align="center">

# 3DL-Net

### Dilated Dendritic Learning for Medical Image Segmentation

**Global context · Local detail · Dendritic learning**

[![Paper](https://img.shields.io/badge/Paper-ESWA%202025-3776AB?style=flat-square)](https://doi.org/10.1016/j.eswa.2024.125874)
[![Open Access](https://img.shields.io/badge/Open_Access-Yes-2EA44F?style=flat-square)](https://doi.org/10.1016/j.eswa.2024.125874)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)

[Paper](https://doi.org/10.1016/j.eswa.2024.125874) · [Setup](#setup) · [Data](#data-preparation) · [Training](#training) · [Citation](#citation)

</div>

## Overview

3DL-Net combines multi-scale dilated convolutions for global context with dendritic neuron modules for local feature refinement, trained with deep supervision. This repository contains the PyTorch model implementation, four-fold training and comparison scripts, and prediction/visualization utilities.

**Paper:** *Dilated dendritic learning of global–local feature representation for medical image segmentation*  
**Authors:** Zhipeng Liu, Yaotong Song, Junyan Yi, Zhiming Zhang, Masaaki Omura, Zhenyu Lei and Shangce Gao  
**Publication:** *Expert Systems with Applications*, Volume 264, Article 125874, 2025  
**DOI:** [10.1016/j.eswa.2024.125874](https://doi.org/10.1016/j.eswa.2024.125874)

## Method at a Glance

The diagram is a conceptual overview. See the paper for the full architecture and [`model_net.py`](model_net.py) for the implementation (`RRCNet`).

```mermaid
flowchart LR
    A[Medical image] --> B[DSNet: preliminary segmentation]
    B --> C[DMNet: missed-region refinement]
    C --> D[Segmentation output]
    E[Dendritic neuron modules] -. Local refinement .-> B
    E -. Local refinement .-> C
```

- **DSNet:** encoder–decoder segmentation with deep supervision.
- **DMNet:** multi-scale dilated convolutions for contextual refinement.
- **DNM:** dendritic learning for local feature representation.

## Setup

```bash
git clone https://github.com/lzp990616/3DL-Net.git
cd 3DL-Net
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in Windows PowerShell. Install mutually compatible PyTorch/torchvision versions for your hardware, then the packages imported by the main training path:

```bash
python -m pip install torch torchvision
python -m pip install numpy Pillow opencv-python scikit-learn matplotlib tqdm torchmetrics MedPy torchsummary torchstat
```

**Compatibility:** the main training script has no pinned environment. It imports legacy TorchMetrics APIs including `dice_score` and `precision_recall`; newer releases may need import/API changes. The commands list dependencies, not a verified version lock. Comparison models also import packages such as `einops`, `timm` and `fairscale`.

## Data Preparation

Obtain datasets separately using the sources and access conditions cited in the paper. For example:

```text
3DL-Net/
└── data/
    ├── STU-Hospital-master/
    │   └── data_mask/
    │       ├── images/
    │       └── masks/
    └── BUS/
        └── BUS/
            └── data_mask/
                ├── images/
                └── masks/
```

The four-fold loader sorts image and mask filenames separately and pairs them by index. Ensure equal file counts and corresponding sorted filenames. The main training script loads images as RGB and masks as grayscale, resizing both to 384 × 384.

### Dataset selectors

These are the branches in `k_folder.py`, rather than a list of the paper's evaluation datasets:

| `--data_name` | Default root path |
| --- | --- |
| `stu` | `./data/STU-Hospital-master/` |
| `bus` | `./data/BUS/BUS/` |
| `busi` | `./data/Dataset_BUSI/Dataset_BUSI_with_GT/` |
| `polyp` | `./data/Kvasir-SEG/` |
| `brain` | `./data/brain_tumor/` |

Each root must contain `data_mask/images/` and `data_mask/masks/`. Edit path constants if your layout differs. The paper also evaluates COVID-19 data, but `k_folder.py` currently has no `covid` branch; configure one before training on that dataset.

### Optional directory-based split

`k_folder.py` reads `data_mask/` directly and creates four folds internally; it does **not** require `data_spilt.py` first.

Prediction utilities instead read `data/train/` and `data/val/`. If using `data_spilt.py`, set its `filepath` to the intended dataset root with a trailing slash. Current defaults are 80% training, 20% validation and 0% test. The script **deletes and recreates** the six split directories under that root; preserve any split you need before running it.

```bash
python data_spilt.py
```

## Training

Create the log directory before running the script:

```bash
python -c "from pathlib import Path; Path('log').mkdir(exist_ok=True)"
python k_folder.py --data_name stu --log_name ./log/3dlnet_stu.log --EPOCH 100 --LR 0.00005 --LOSSK 0.1 --loss_name dynamic_focal --DNM 1 --DNM_MR 1 --M 10 --flag_deeplab 1 --DEVICE 0
```

The script uses four-fold cross-validation with shuffled splits and Adam optimization. Logs include training loss, Dice, Jaccard and other validation metrics.

| Argument | Default | Meaning |
| --- | --- | --- |
| `--data_name` | `stu` | Dataset selector above |
| `--EPOCH` | `100` | Epochs per fold |
| `--LR` | `0.00005` | Learning rate |
| `--LOSSK` | `0.1` | Weight on auxiliary loss terms |
| `--loss_name` | `dynamic_focal` | `dynamic_focal`, `focal` or `dice` |
| `--M` | `10` | Number of dendrites |
| `--DNM` | `1` | Dendritic module flag passed to `RRCNet` |
| `--DNM_MR` | `1` | Refinement-branch dendritic module flag |
| `--flag_deeplab` | `1` | Dilated-context branch flag |
| `--DEVICE` | `0` | `0` selects `cuda:0`; other values select `cuda:1` when CUDA is available |
| `--log_name` | `./log/test3_8.log` | Log path; use a different name per run |

### Implementation notes

- Main training DataLoaders use a fixed batch size of **6** and `drop_last=True`. The parsed `--batch_size` option does not override these values.
- `KFold` has no fixed `random_state`, so repeated runs can use different splits. Preserve splits and environment settings when comparing experiments.
- Checkpoint-saving code in `k_folder.py` is commented out, and its parsed `--ckpt` option is not wired into loading. Configure saving/loading if trained weights are needed.
- These instructions describe the checked-in code; they do not claim a fresh end-to-end reproduction of the paper's results. Refer to the paper for reported metrics and the experimental protocol.

## Comparison Experiments

`compare_kfold.py` provides baseline experiments. Check its model imports and dataset branches before running; they differ from the main script. Example with U-Net on STU:

```bash
python compare_kfold.py --data_name stu --model_name unet --log_name ./log/unet_stu.log --EPOCH 100 --LR 0.00005 --DEVICE 0
```

The comparison DataLoaders currently use a fixed batch size of 12; their parsed `--batch_size` option does not override it. Use matching data, folds and evaluation settings for meaningful comparisons.

## Prediction and Visualization

Prediction scripts require a compatible trained checkpoint and dataset split. Review their script-level paths before running:

| Script | Configuration to check |
| --- | --- |
| `predicted.py` | Dataset selector, `pretrained_model` (currently `./pic/model/dnm_bus.pth`), model flags and output directories |
| `predicted_other_model.py` | Model selection, dataset path (currently COVID-19), checkpoint path and output directories |
| `pic_*.py` | Paths for figure composition, ablations and feature visualization |

The prediction loaders expect `data/train`, `data/trainannot`, `data/val` and `data/valannot` under the dataset root. Checkpoints are loaded from a dictionary containing `model_static_dict`. Update the script-level checkpoint path; passing `--ckpt` alone does not replace it. Create the output directories referenced by the script before exporting images.

```bash
# After configuring data, checkpoint and output paths:
python predicted.py --data_name bus --log_name ./log/predict_bus.log
```

No pretrained `.pth` or `.pt` weights are included in this checkout.

## Repository Guide

| File or directory | Purpose |
| --- | --- |
| `model_net.py` | 3DL-Net implementation (`RRCNet`) |
| `conv_dnm.py` | Dendritic neuron modules |
| `dataset.py` | Dataset classes and image/mask loading |
| `k_folder.py` | Main four-fold training and evaluation |
| `compare_kfold.py` | Baseline comparison experiments |
| `data_spilt.py` | Optional directory-based split |
| `predicted*.py` | Prediction and image export |
| `pic_*.py` | Figure and feature visualization |
| `model/`, `networks/` | Comparison-model implementations |

## Citation

If this work is useful to your research, please cite:

```bibtex
@article{liu2025dilated,
  title={Dilated dendritic learning of global--local feature representation for medical image segmentation},
  author={Liu, Zhipeng and Song, Yaotong and Yi, Junyan and Zhang, Zhiming and Omura, Masaaki and Lei, Zhenyu and Gao, Shangce},
  journal={Expert Systems with Applications},
  volume={264},
  pages={125874},
  year={2025},
  doi={10.1016/j.eswa.2024.125874},
  publisher={Elsevier}
}
```

## License and Third-Party Code

This checkout has no top-level license file. Bundled third-party components include their own license files; see those files and the original projects for their terms.

## Questions

For code or reproducibility questions, please [open an issue](https://github.com/lzp990616/3DL-Net/issues) with your environment, command and error message.
