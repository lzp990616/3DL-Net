# 3DL-Net: Dilated Dendritic Learning for Medical Image Segmentation

[![Paper](https://img.shields.io/badge/Paper-Expert%20Systems%20with%20Applications-orange)](https://doi.org/10.1016/j.eswa.2024.125874)
[![Access](https://img.shields.io/badge/Open%20Access-green)](https://doi.org/10.1016/j.eswa.2024.125874)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-red)](https://pytorch.org/)

> **Dilated dendritic learning of global–local feature representation for medical image segmentation**
> [Zhipeng Liu](https://scholar.google.com/citations?hl=zh-CN&user=2r4K18sAAAAJ), Yaotong Song, Junyan Yi, Zhiming Zhang, Masaaki Omura, [Zhenyu Lei](https://scholar.google.com/citations?user=7Ss6peAAAAAJ&hl=zh-CN&oi=sra), [Shangce Gao](https://toyamaailab.github.io/)
>
> *Expert Systems with Applications, Vol. 264, 2025* (Open Access) · [📄 Paper](https://doi.org/10.1016/j.eswa.2024.125874)

3DL-Net is a medical image segmentation network that jointly models **global** and **local** feature representations. It combines dilated convolutions (for a larger receptive field and richer global context) with a **dendritic neuron module (DNM)** that refines local fine-grained features, trained under a deep-supervision scheme with a customized loss.

## Highlights

- **Global feature enhancement** via dilated convolutions with a pyramid structure, capturing multi-scale contextual information.
- **Dendritic neuron module (DNM) at the channel level** — to the best of our knowledge, the first use of dendritic learning at the channel level for refining local features.
- **Deep supervision** with a self-adaptive focal + BCE loss to handle class imbalance and sharpen boundaries.
- Evaluated on three public datasets (**BUS**, **STU**, **COVID-19**) with k-fold cross-validation, outperforming state-of-the-art methods on multiple metrics.

## Method Overview

3DL-Net consists of three components:

1. **DSNet** — preliminary segmentation with deep supervision (SegNet-based encoder–decoder).
2. **DMNet** — multi-scale contextual module based on dilated convolutions and a pyramid structure, for detecting missed lesion regions and enhancing global features.
3. **DNM** — a dendritic neuron module cascaded at the end of DSNet and DMNet to refine shallow/local features.

## Requirements

```bash
python >= 3.8
torch >= 1.10
torchvision
numpy
opencv-python
scikit-learn
matplotlib
```

```bash
pip install torch torchvision numpy opencv-python scikit-learn matplotlib
```

## Dataset Structure

```
├── data
│   ├── BUS
│   ├── STU-Hospital-master
│   └── COVID-19
│
│   └── <dataset>/
│       ├── data_mask
│       │   ├── images
│       │   └── masks
│       └── data            # only need to create `data` and `data_mask`
│           ├── train
│           ├── trainannot
│           ├── val
│           ├── valannot
│           ├── test
│           └── testannot
```

## Getting Started

### 1. Data preprocessing

```bash
python data_spilt.py    # split the dataset; set 'filepath' inside the script
```

### 2. K-fold experiment (3DL-Net)

```bash
python k_folder.py --log_name "./log/log_name.log" --data_name stu \
       --batch_size 8 --EPOCH 100 --LR 0.00005 --LOSSK 0.1 \
       --DNM 1 --DNM_MR 1 --M 10 --flag_deeplab 1
```

### 3. Comparative experiments

```bash
python compare_kfold.py --log_name "./log/log_name.log" --data_name stu \
       --batch_size 8 --EPOCH 100 --LR 0.00005
```

### 4. Prediction / visualization

```bash
python predicted.py
python predicted_other_model.py
```

### Key Arguments

| Argument | Meaning |
| --- | --- |
| `--data_name` | dataset name (e.g. `stu`, `bus`, `covid`) |
| `--batch_size` | training batch size |
| `--EPOCH` | number of training epochs |
| `--LR` | learning rate |
| `--LOSSK` | weight `k` balancing BCE and adaptive focal loss |
| `--DNM` | enable (1) / disable (0) the dendritic neuron module |
| `--DNM_MR` | enable (1) / disable (0) DNM in the missed-region (DMNet) branch |
| `--M` | number of dendrites in the DNM |
| `--flag_deeplab` | enable (1) / disable (0) the dilated (DeepLab-style) branch |

## Repository Structure

```
3DL-Net_code/
├── k_folder.py              # main k-fold training of 3DL-Net
├── compare_kfold.py         # k-fold training of comparison models
├── conv_dnm.py              # dendritic neuron module (DNM)
├── model_net.py             # 3DL-Net network definition
├── dataset.py               # dataset / dataloader
├── data_spilt.py            # data splitting
├── predicted*.py            # inference & visualization
└── model/                   # baseline models (U-Net, TransUNet, SegNet, BRAU-Net++, ...)
```

## Citation

```bibtex
@article{liu2025dilated,
  title={Dilated dendritic learning of global--local feature representation for medical image segmentation},
  author={Liu, Zhipeng and Song, Yaotong and Yi, Junyan and Zhang, Zhiming and Omura, Masaaki and Lei, Zhenyu and Gao, Shangce},
  journal={Expert Systems with Applications},
  volume={264},
  pages={125874},
  year={2025},
  publisher={Elsevier}
}
```
