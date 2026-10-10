# continual-learning-wildlife
This repository contains the official Python implementation for the paper currently under review at *Methods in Ecology and Evolution*. 

To protect the double-blind review process, the authors' identities and institutional affiliations have been thoroughly anonymized across all related platforms.

---

## 🛠️ Environment & Installation

This project is implemented in **Python 3.11.11**. We recommend using a virtual environment (e.g., `conda` or `venv`) to manage the dependencies.

### Key Prerequisites

| Package | Version | Description |
| :--- | :--- | :--- |
| `torch` | `2.6.0+cu124` | PyTorch with CUDA 12.4 support |
| `torchvision` | `0.21.0+cu124` | Computer Vision utilities |
| `transformers` | `4.30.0` | Hugging Face Transformers |
| `scikit-learn` | `1.7.1` | Machine learning utilities |
| `numpy` | `2.0.1` | Numerical computation |
| `tensorboard` | `2.19.0` | Training visualization |
| `tqdm` | `4.67.1` | Progress bar |
---
# 1. Create and activate a Python 3.11.11 virtual environment (using Conda as an example)
`conda create -n wildlife_cl python=3.11.11 -y`

`conda activate wildlife_cl`

# 2. Install all dependencies with a single command
`pip install -r requirements.txt`

# 1. 建立並啟用 Python 3.11.11 的虛擬環境 (以 Conda 為例)
`conda create -n wildlife_cl python=3.11.11 -y`

`conda activate wildlife_cl`

# 2. 安裝所有依賴函式庫
`pip install -r requirements.txt`
