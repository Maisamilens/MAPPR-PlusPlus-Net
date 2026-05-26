
```markdown
# 🔥 MAPPR++-Net: Multi-Path Attention and Progressive Perception Refinement Plus Plus Network for Infrared Small Target Detection

[![Paper](https://img.shields.io/badge/Paper-IEEE-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-orange)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Official implementation of **MAPPR++-Net** — a novel deep learning framework for Infrared Small Target Detection (ISTD) that achieves state-of-the-art performance on NUAA-SIRST, IRSTD-1k, and NUDT-SIRST datasets.

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Architecture Details & Module References](#-architecture-details--module-references)
- [Dataset Preparation](#-dataset-preparation)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Acknowledgments](#-acknowledgments)
- [Citation](#-citation)

---

## 🌟 Overview

Robust detection of diminutive infrared signatures represents a cornerstone capability for contemporary surveillance systems, catastrophe early-warning platforms, and precision-guided weaponry. Prevailing algorithms demonstrate critical performance degradation when confronting adverse operational conditions characterized by suppressed signal-to-noise ratios, ambiguous target geometries, and dense background interference.

**MAPPR++-Net** introduces an integrated detection framework optimized for accurate ISTD in complex backgrounds:
1. **STKE & KAFF**: Strengthen shallow key features and sharpen blurred edges of small infrared targets masked by background noise.
2. **CFPR**: Iterative fusion of low-level and deep semantic cues via Lightweight Residual Self-Attention (LRSA) and RepC3 blocks to improve robustness.
3. **MARF++ Block**: Enhanced Multi-path Attention Residual Fusion with Dynamic Path Selection (DPS), Cross-Path Interaction (CPI), and Multi-Scale Feature Pyramid (MSFP).
4. **AGRM & CARM**: Dual-level hierarchical refinement modules for middle and high-level feature enhancement.

---

## 📐 Architecture Details & Module References

Our network extends the **RT-DETR** backbone. The overall architecture and specific module designs are inspired by and built upon the following foundational papers and concepts:

### 1. Shallow Target Knowledge Enhancement (STKE)
Designed as a plug-and-play component after Stage 1 (S1). It uses three parallel convolutional streams (1×1, 3×3, dilated 3×3) followed by channel attention to refine low-level features.
*   **Inspiration**: Knowledge Enhancement and Clinical Pathway Encoding strategies.
*   **Reference Paper**: Wu, J., Wu, X., Zheng, Y., & Yang, J. (2024). *MedKP: Medical Dialogue with Knowledge Enhancement and Clinical Pathway Encoding*.
*   **Link**: [https://arxiv.org/pdf/2403.06611](https://arxiv.org/pdf/2403.06611)

### 2. Key-Aware Feature Fusion (KAFF)
Employs a dual-branch gated fusion mechanism between the original shallow features and the STKE-enhanced features. It dynamically controls the fusion ratio to preserve salient target information and suppress background noise.
*   **Inspiration**: Key-Aware Feature Fusion mechanisms for adaptive weighting.
*   **Reference Paper**: IEEE Xplore (2019). *Key-Aware Feature Fusion*.
*   **Link**: [https://ieeexplore.ieee.org/abstract/document/8732991](https://ieeexplore.ieee.org/abstract/document/8732991)

### 3. Cross-Feature Progressive Refinement (CFPR)
Progressively refines hierarchical features by merging shallow and deep layers through LRSA (Lightweight Residual Self-Attention) and RepC3 fusion blocks, achieving stronger target consistency.
*   **Inspiration**: Cross-Feature Progressive Refinement and structural re-parameterization.
*   **Reference Paper**: arXiv (2023). *Cross-Feature Progressive Refinement*.
*   **Link**: [https://arxiv.org/pdf/2311.04625](https://arxiv.org/pdf/2311.04625)

### 4. Multi-path Attention Residual Fusion Plus Plus (MARF++)
The core backbone block replacing standard ResNet BasicBlocks. It features three complementary attention pathways (Channel-Spatial cascade, Channel Non-Local, and Lightweight Transformer-Assisted) unified via Dynamic Path Selection (DPS) and Cross-Path Interaction (CPI).
*   **Inspiration**: Multi-path Attention Residual Fusion.
*   **Reference Paper**: ScienceDirect (2025). *Multi-path Attention Residual Fusion*.
*   **Link**: [https://www.sciencedirect.com/science/article/pii/S1532046425001947](https://www.sciencedirect.com/science/article/pii/S1532046425001947)

### 5. Adaptive Global Refinement Module (AGRM)
Applied at Stage S2 to concentrate computational resources on target-specific middle-level regions using sequential spatial and channel attention.
*   **Inspiration**: Adaptive Global Refinement Module.
*   **Reference Paper**: IEEE Xplore (2022). *Adaptive Global Refinement Module*.
*   **Link**: [https://ieeexplore.ieee.org/abstract/document/9624979](https://ieeexplore.ieee.org/abstract/document/9624979)

### 6. Context-Aware Refinement Module (CARM)
Applied at Stage S3 to capture expansive long-range dependencies within deep feature spaces using Atrous Spatial Pyramid Pooling (ASPP).
*   **Inspiration**: Context-Aware Refinement and ASPP (DeepLab v3).
*   **ASPP Reference**: Chen, L.C., et al. (2017). *DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs*. [arXiv:1606.00915](https://arxiv.org/abs/1606.00915)

### 7. Base Framework: RT-DETR
The overall detection pipeline is built upon the Real-Time DEtection TRansformer (RT-DETR) architecture, utilizing a Hybrid Encoder and Transformer Decoder.
*   **Reference Paper**: Zhao, Y., et al. (2024). *DETRs Beat YOLOs on Real-Time Object Detection*. CVPR 2024.
*   **Link**: [https://arxiv.org/abs/2304.08069](https://arxiv.org/abs/2304.08069)

---

## 📦 Dataset Preparation

This project supports the **IRSTD-1k** dataset. By default, the config is set up for the following local directory structure:

```text
D:\PhD Publications\Target Detection\data\IRSTD-1k\IRSTD-1k\
├── IRSTD1k_Img\          # Infrared images (XDU0.png, XDU1.png, ...)
├── IRSTD1k_Label\        # Binary mask labels (XDU0.png, XDU1.png, ...)
├── labels\               # (Generated) YOLO format bounding box labels
├── trainval.txt          # Train + Validation split list
├── test.txt              # Test split list
└── trainvaltest.txt      # Full dataset list
```

### Step 1: Generate YOLO-format Bounding Box Labels
Since the IRSTD-1k dataset provides pixel-level segmentation masks but our detection model requires bounding boxes, run the provided preparation script to extract YOLO-format labels from the masks:

```bash
python datasets/irstd1k_dataset.py --prepare_labels \
    --img_dir "D:\PhD Publications\Target Detection\data\IRSTD-1k\IRSTD-1k\IRSTD1k_Img" \
    --mask_dir "D:\PhD Publications\Target Detection\data\IRSTD-1k\IRSTD-1k\IRSTD1k_Label" \
    --output_dir "D:\PhD Publications\Target Detection\data\IRSTD-1k\IRSTD-1k\labels"
```

### Step 2: Data Splitting
The dataloader automatically handles the 6:2:2 (Train:Val:Test) splitting logic based on the `trainval.txt` and `test.txt` files located in the dataset root.

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/Maisamilens/MAPPR-PlusPlus-Net.git
cd MAPPR-PlusPlus-Net

# Create conda environment (optional but recommended)
conda create -n mapprpp python=3.10 -y
conda activate mapprpp

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

All configurations are managed via `configs/mapprpp_irstd1k.yaml`. Ensure the dataset paths match your local setup.

### Training
Train the full MAPPR++-Net model for 300 epochs:
```bash
python train.py --config configs/mapprpp_irstd1k.yaml
```
*Checkpoints and logs will be saved to `checkpoints/` and `logs/` respectively.*

### Testing
Evaluate the model on the test set:
```bash
python test.py --config configs/mapprpp_irstd1k.yaml \
    --checkpoint checkpoints/best_model.pth \
    --split test \
    --save_results
```

### Prediction
Run inference on custom images or the full image directory:
```bash
python predict.py --config configs/mapprpp_irstd1k.yaml \
    --checkpoint checkpoints/best_model.pth \
    --input_dir "D:\PhD Publications\Target Detection\data\IRSTD-1k\IRSTD-1k\IRSTD1k_Img" \
    --output_dir predictions/
```

### Ablation Study
Reproduce the ablation studies from the paper (Tables V, VI, VII, VIII):
```bash
python ablation_study.py --config configs/mapprpp_irstd1k.yaml \
    --output_dir runs/ablation/
```

### Visualization
Generate paper figures (Radar charts, 3D confidence bars, detection maps):
```bash
python visualize_results.py --config configs/mapprpp_irstd1k.yaml \
    --checkpoint checkpoints/best_model.pth \
    --output_dir visualizations/
```

---

## 📊 Results

Our method achieves state-of-the-art performance on standard IRSTD benchmarks.

### IRSTD-1k Dataset
| Method | Precision (%) | Recall (%) | F1 (%) |
|--------|---------------|------------|--------|
| DNANet | 76.8 | 72.1 | 74.4 |
| EFLNet | 87.0 | 81.7 | 84.3 |
| STASPPNet | 86.4 | 79.8 | 83.0 |
| **MAPPR++-Net (Ours)** | **88.9** | **84.2** | **86.2** |

### NUAA-SIRST Dataset
| Method | mAP50 (%) | mAP50-95 (%) | Precision (%) | Recall (%) |
|--------|-----------|--------------|---------------|------------|
| RT-DETR | 91.8 | 45.6 | 96.3 | 84.6 |
| STASPPNet | 95.2 | 48.6 | 95.0 | 92.9 |
| **MAPPR++-Net (Ours)** | **95.2** | **49.1** | **97.8** | **90.8** |

---

## 🙏 Acknowledgments

This research and codebase builds upon the foundational work of many outstanding researchers in the infrared small target detection and computer vision communities. We extend our deepest gratitude to the following:

### Special Thanks to Key GitHub Contributors
*   **[XinyiYing/LESPS](https://github.com/XinyiYing/LESPS)** — Repository for *"Mapping Degeneration Meets Label Evolution: Learning Infrared Small Target Detection with Single Point Supervision"* (CVPR 2023). Their pioneering work on label evolution and single-point supervision has significantly advanced weakly-supervised IRSTD research and informed our understanding of target degeneration mapping.
*   **[YeRen123455/ICPR-Track2-LightWeight](https://github.com/YeRen123455/ICPR-Track2-LightWeight)** — Lightweight infrared small target detection framework. Their contributions to efficient and lightweight architectures for IRSTD have informed our design choices for computational efficiency in progressive refinement.

### Frameworks and Base Implementations
*   **[lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)** — For the excellent Real-Time DETR implementation that serves as our base detection framework.
*   **[RuiZhang97/ISNet](https://github.com/RuiZhang97/ISNet)** — For providing the IRSTD-1k dataset and the ISNet baseline, which is crucial for benchmarking ISTD models.
*   **[meituan/YOLOv6](https://github.com/meituan/YOLOv6)** — For the inspiration behind the RepC3 re-parameterization block used in our CFPR module.

### Dataset Providers
*   **NUAA-SIRST**: Dai et al. for the Asymmetric Contextual Modulation (ACM) dataset.
*   **IRSTD-1k**: Zhang et al. for the ISNet dataset.
*   **NUDT-SIRST**: Li et al. for the Dense Nested Attention Network (DNANet) dataset.

### Module Architecture Inspirations
We sincerely thank the authors of the following works whose concepts were instrumental in designing the modules of MAPPR++-Net:
*   Wu et al. for the **STKE** concept ([arXiv:2403.06611](https://arxiv.org/pdf/2403.06611))
*   The authors of the **KAFF** mechanism ([IEEE:8732991](https://ieeexplore.ieee.org/abstract/document/8732991))
*   The authors of the **CFPR** approach ([arXiv:2311.04625](https://arxiv.org/pdf/2311.04625))
*   The authors of the **AGRM** module ([IEEE:9624979](https://ieeexplore.ieee.org/abstract/document/9624979))
*   The authors of the **MARF** methodology ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1532046425001947))

---

## 📝 Citation

If you find this work or code useful for your research, please consider citing our paper:

```bibtex
@article{abbas2025mapprpp,
  title={MAPPR++-Net: Multi-Path Attention and Progressive Perception Refinement Plus Plus Network for Infrared Small Target Detection},
  author={Abbas, Maisam and Hussain, Muhammad and Ali, Qaisar and Hassan, Muhammad and Wang, Ran-Zan},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2025}
}
```

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
```
