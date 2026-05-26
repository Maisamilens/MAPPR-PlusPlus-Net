# MAPPR++-Net

## Multi-Path Attention and Progressive Perception Refinement Plus Plus Network for Infrared Small Target Detection

![Python](https://img.shields.io/badge/Python-3.10+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Framework](https://img.shields.io/badge/Framework-RT--DETR-blue.svg)

---

## Overview

Infrared Small Target Detection (ISTD) is an important research area for:

- Surveillance systems
- Aerospace monitoring
- Maritime target tracking
- Early-warning systems
- Remote sensing applications

However, ISTD remains challenging because of:

- Extremely small target sizes
- Low signal-to-noise ratio
- Cluttered backgrounds
- Weak thermal signatures
- Limited texture information

MAPPR++-Net is a deep learning framework designed for robust and accurate infrared small target detection in complex environments.

The framework integrates:

- Multi-path attention learning
- Progressive feature refinement
- Dynamic feature fusion
- Transformer-assisted perception
- Cross-scale semantic enhancement

MAPPR++-Net achieves strong performance on:

- IRSTD-1k
- NUAA-SIRST
- NUDT-SIRST

datasets.

---

## Key Contributions

### STKE + KAFF Modules

Enhance shallow target representations while suppressing background interference.

### CFPR Framework

Progressively refines semantic and low-level features using:

- LRSA attention
- RepC3 fusion
- Residual refinement

### MARF++ Block

Introduces:

- Dynamic Path Selection (DPS)
- Cross-Path Interaction (CPI)
- Multi-Scale Feature Pyramid (MSFP)

for improved feature extraction.

### AGRM + CARM

Dual-level refinement modules improve:

- Middle-level contextual representation
- High-level semantic consistency

---

## Network Architecture

The proposed MAPPR++-Net extends the RT-DETR backbone with several custom-designed modules.

---

## 1. Shallow Target Knowledge Enhancement (STKE)

A plug-and-play enhancement module inserted after Stage-1.

### Features

- Parallel convolution branches
- Dilated convolutions
- Channel attention refinement

### Purpose

Enhance weak infrared target structures while preserving fine details.

### Reference

Wu, J., Wu, X., Zheng, Y., & Yang, J. (2024).  
MedKP: Medical Dialogue with Knowledge Enhancement and Clinical Pathway Encoding.

https://arxiv.org/pdf/2403.06611

---

## 2. Key-Aware Feature Fusion (KAFF)

A dual-branch adaptive fusion mechanism.

### Features

- Dynamic gating
- Saliency-aware weighting
- Noise suppression

### Purpose

Preserve discriminative infrared target information.

### Reference

https://ieeexplore.ieee.org/abstract/document/8732991

---

## 3. Cross-Feature Progressive Refinement (CFPR)

A hierarchical feature refinement strategy.

### Components

- Lightweight Residual Self-Attention (LRSA)
- RepC3 blocks
- Progressive semantic fusion

### Purpose

Improve target consistency across scales.

### Reference

https://arxiv.org/pdf/2311.04625

---

## 4. MARF++ Block

### Multi-path Attention Residual Fusion Plus Plus

Core backbone replacement for conventional residual blocks.

### Components

- Channel-Spatial Attention
- Channel Non-Local Attention
- Lightweight Transformer Pathway
- Dynamic Path Selection (DPS)
- Cross-Path Interaction (CPI)
- Multi-Scale Feature Pyramid (MSFP)

### Purpose

Capture long-range dependencies and complementary contextual cues.

### Reference

https://www.sciencedirect.com/science/article/pii/S1532046425001947

---

## 5. Adaptive Global Refinement Module (AGRM)

Applied at Stage S2.

### Purpose

Enhance middle-level semantic representations using:

- Spatial attention
- Channel refinement

### Reference

https://ieeexplore.ieee.org/abstract/document/9624979

---

## 6. Context-Aware Refinement Module (CARM)

Applied at Stage S3.

### Features

- Atrous Spatial Pyramid Pooling (ASPP)
- Multi-scale receptive field aggregation

### Purpose

Capture global contextual information for robust target localization.

### Reference

https://arxiv.org/abs/1606.00915

---

## 7. RT-DETR Backbone

MAPPR++-Net is built upon the RT-DETR framework.

### Features

- Hybrid encoder
- Transformer decoder
- Real-time detection capability

### Reference

https://arxiv.org/abs/2304.08069

---

## Dataset Preparation

This implementation supports the IRSTD-1k dataset.

---

## Expected Directory Structure

```text
D:\PhD Publications\Target Detection\data\IRSTD-1k\IRSTD-1k\
├── IRSTD1k_Img\
├── IRSTD1k_Label\
├── labels\
├── trainval.txt
├── test.txt
└── trainvaltest.txt
```

---

## Step 1 — Generate YOLO Bounding Box Labels

Since IRSTD-1k provides segmentation masks, convert masks into YOLO-format annotations:

```bash
python datasets/irstd1k_dataset.py --prepare_labels \
    --img_dir "D:\PhD Publications\Target Detection\data\IRSTD-1k\IRSTD-1k\IRSTD1k_Img" \
    --mask_dir "D:\PhD Publications\Target Detection\data\IRSTD-1k\IRSTD-1k\IRSTD1k_Label" \
    --output_dir "D:\PhD Publications\Target Detection\data\IRSTD-1k\IRSTD-1k\labels"
```

---

## Step 2 — Dataset Splitting

The dataloader automatically performs:

- Train : Validation : Test = 6 : 2 : 2

using:

- `trainval.txt`
- `test.txt`

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Maisamilens/MAPPR-PlusPlus-Net.git
cd MAPPR-PlusPlus-Net
```

---

### Create Environment

```bash
conda create -n mapprpp python=3.10 -y
conda activate mapprpp
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

All configurations are managed through:

```text
configs/mapprpp_irstd1k.yaml
```

---

## Training

Train MAPPR++-Net for 300 epochs:

```bash
python train.py --config configs/mapprpp_irstd1k.yaml
```

Outputs:

- Checkpoints → `checkpoints/`
- Logs → `logs/`

---

## Testing

Evaluate on the test split:

```bash
python test.py --config configs/mapprpp_irstd1k.yaml \
    --checkpoint checkpoints/best_model.pth \
    --split test \
    --save_results
```

---

## Prediction

Run inference on infrared images:

```bash
python predict.py --config configs/mapprpp_irstd1k.yaml \
    --checkpoint checkpoints/best_model.pth \
    --input_dir "D:\PhD Publications\Target Detection\data\IRSTD-1k\IRSTD-1k\IRSTD1k_Img" \
    --output_dir predictions/
```

---

## Ablation Study

Reproduce ablation experiments:

```bash
python ablation_study.py --config configs/mapprpp_irstd1k.yaml \
    --output_dir runs/ablation/
```

---

## Visualization

Generate:

- Detection maps
- Comparative plots
- Confidence visualizations
- Performance charts

```bash
python visualize_results.py --config configs/mapprpp_irstd1k.yaml \
    --checkpoint checkpoints/best_model.pth \
    --output_dir visualizations/
```

---

## Experimental Results

### IRSTD-1k Dataset

| Method | Precision (%) | Recall (%) | F1 (%) |
|---|---|---|---|
| DNANet | 76.8 | 72.1 | 74.4 |
| EFLNet | 87.0 | 81.7 | 84.3 |
| STASPPNet | 86.4 | 79.8 | 83.0 |
| MAPPR++-Net (Ours) | 88.9 | 84.2 | 86.2 |

---

### NUAA-SIRST Dataset

| Method | mAP50 (%) | mAP50-95 (%) | Precision (%) | Recall (%) |
|---|---|---|---|---|
| RT-DETR | 91.8 | 45.6 | 96.3 | 84.6 |
| STASPPNet | 95.2 | 48.6 | 95.0 | 92.9 |
| MAPPR++-Net (Ours) | 95.2 | 49.1 | 97.8 | 90.8 |

---

## Acknowledgments

This work builds upon several important open-source repositories and research contributions.

### Repositories

LESPS  
https://github.com/XinyiYing/LESPS

ICPR Lightweight Track  
https://github.com/YeRen123455/ICPR-Track2-LightWeight

RT-DETR  
https://github.com/lyuwenyu/RT-DETR

ISNet  
https://github.com/RuiZhang97/ISNet

YOLOv6  
https://github.com/meituan/YOLOv6

---

## Citation

```bibtex
@article{abbas2025mapprpp,
  title={MAPPR++-Net: Multi-Path Attention and Progressive Perception Refinement Plus Plus Network for Infrared Small Target Detection},
  author={Abbas, Maisam and Hussain, Muhammad and Ali, Qaisar and Hassan, Muhammad and Wang, Ran-Zan},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2025}
}
```

---

## License

This project is released under the MIT License.

See:

```text
LICENSE
```

for additional details.

---

## Contact

Maisam Abbas  
Department of Computer Science and Engineering  
Yuan Ze University, Taiwan  

Email: s1129105@mail.yzu.edu.tw
