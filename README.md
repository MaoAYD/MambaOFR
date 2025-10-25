# [CVPR2025] Making Old Film Great Again: Degradation-aware State Space Model for Old Film Restoration

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)


## 📖 Overview

This repository contains the implementation of MambaOFR for old film restoration. 

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/MaoAYD/MambaOFR.git
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📊 Dataset Preparation

### Downloading Datasets

- **Traning Data**

    - REDS (Type:sharp) [Download](https://seungjunnah.github.io/Datasets/reds.html)

      Download the relabled scratch templates for the video degradation model. [Download](https://portland-my.sharepoint.com/personal/ziyuwan2-c_my_cityu_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fziyuwan2%2Dc%5Fmy%5Fcityu%5Fedu%5Fhk%2FDocuments%2FCVPR2022%2Ftexture%5Ftemplate%2Ezip&parent=%2Fpersonal%2Fziyuwan2%2Dc%5Fmy%5Fcityu%5Fedu%5Fhk%2FDocuments%2FCVPR2022&ga=1) and put it on the root directory.

      ```
      MambaOFR
      ├── VP_code
      ├── noise_data
      ```

- **Test Data**

    - Degraded REDS val [Download](https://drive.google.com/file/d/1LCSyr0fJGSKi_NZ6--rbniaO1n59pGBO/view?usp=sharing)

    - Real-world Old film Dataset [Download](https://drive.google.com/file/d/1Ob7efeBKaVY0cQ79O_dI-BLe4152kDUz/view?usp=sharing)
 
    - Degrade your own data
      ```bash
      python degradation.py --input input_dir --output output_dir
      ```


## 🏋️ Training

```bash
ToDO Coming Soon
```


## 📈 Evaluation

### Test on Validation Set

```
ToDO Coming Soon
```

## 🎯 Inference

### Quick Inference

```bash
ToDO Coming Soon
```

## 📜 Citation

If you use this code in your research, please cite:

```bibtex
@INPROCEEDINGS{11093047,
  author={Mao, Yudong and Luo, Hao and Zhong, Zhiwei and Chen, Peilin and Zhang, Zhijiang and Wang, Shiqi},
  booktitle={2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={Making Old Film Great Again: Degradation-aware State Space Model for Old Film Restoration}, 
  year={2025},
  volume={},
  number={},
  pages={28039-28049},
  keywords={Degradation;Computer vision;Films;Computational modeling;Video restoration;Benchmark testing;Pattern recognition;Videos;old film restoration;video restoration;dataset},
  doi={10.1109/CVPR52734.2025.02611}}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to [https://github.com/raywzy/Bringing-Old-Films-Back-to-Life] for the excellent deep learning framework
