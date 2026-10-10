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

### 1. Create and activate a Python 3.11.11 virtual environment (using Conda as an example)
`conda create -n wildlife_cl python=3.11.11 -y`

`conda activate wildlife_cl`

### 2. Install all dependencies with a single command
`pip install -r requirements.txt`

### 1. 建立並啟用 Python 3.11.11 的虛擬環境 (以 Conda 為例)
`conda create -n wildlife_cl python=3.11.11 -y`

`conda activate wildlife_cl`

### 2. 安裝所有依賴函式庫
`pip install -r requirements.txt`

---

## File Structure
```text
continual-learning-wildlife/
├── Continual learning/                     # Source code / 原始程式碼
│   ├── Predict_Swin_VIT_ResNet.py          # Swin, ViT, ResNet prediction script / Swin, ViT, ResNet測試程式
│   ├── Predict_ViT-H.py                    # ViT-H prediction script / ViT-H測試程式
│   ├── Train Resnet, Swin, ViT/
│   │   ├── Finetune(CrossEntropy).py       # Main training script / 主要訓練程式
│   │   ├── EWC.py                          # Main training script / 主要訓練程式
│   │   ├── VCL.py                          # Main training script / 主要訓練程式
│   │   └── FIP.py                          # Main training script / 主要訓練程式
│   └── Train ViT-H/
│       ├── Finetune(CrossEntropy).py       # Main training script / 主要訓練程式
│       ├── EWC.py                          # Main training script / 主要訓練程式
│       ├── VCL.py                          # Main training script / 主要訓練程式
│       └── FIP.py                          # Main training script / 主要訓練程式
│
├── Transfer learning/                      # Source code / 原始程式碼
│   ├── Predict_Resnet_Swin_ViT.py          # ResNet, Swin, ViT prediction script / ResNet, Swin, ViT測試程式
│   ├── Predict_ViT-H.py                    # ViT-H prediction script / ViT-H測試程式
│   ├── ResNet-34/
│   │   └── train.py                        # Main training script / 主要訓練程式
│   ├── Swin Transformer/
│   │   └── train.py                        # Main training script / 主要訓練程式
│   ├── ViT/
│   │   └── train.py                        # Main training script / 主要訓練程式
│   └── ViT-H/
│       └── train.py                        # Main training script / 主要訓練程式
│
├── requirements.txt                        # Python dependencies / Python 依賴套件
├── README.md                               # Project documentation / 專案說明文件
└── LICENSE                                 # License file / 授權文件
```

---

## 🔄 Workflow

The overall workflow of this repository follows three main stages:
本專案的整體工作流程分為四個主要階段：

1. Data Preparation / 資料準備
  * Please download the task-specific dataset from Hugging Face: [[hear](https://huggingface.co/Dataset-and-Model)] and place the dataset in the same directory as the training scripts.
  * Experimental result weights can be found [[hear](https://huggingface.co/Dataset-and-Model)].
  * 請到以下hugging face下載任務對應資料集:[[這裡](https://huggingface.co/Dataset-and-Model)]並將資料集與訓練程式放在相同目錄。
  * 實驗結果的權重:[[這裡](https://huggingface.co/Dataset-and-Model)]

2. Model Training / 模型訓練
  * For Continual Learning, use scripts under Continual learning/Train Resnet, Swin, ViT/ and Train ViT-H/ (e.g., EWC.py, VCL.py, FIP.py)
  * For Transfer Learning, use scripts under Transfer learning/ResNet-34/, Swin Transformer/, ViT/, and ViT-H/
  * Continual learning 模式：使用 Continual learning/Train Resnet, Swin, ViT/ 與 Train ViT-H/ 下的程式 (如 EWC.py, VCL.py, FIP.py)
  * Transfer learning 模式：使用 Transfer learning/ResNet-34/, Swin Transformer/, ViT/, ViT-H/ 下的訓練程式

3. Model Evaluation / 模型評估
  * Run prediction scripts (Predict_Swin_VIT_ResNet.py, Predict_ViT-H.py, etc.) to evaluate trained models
  * 執行測試程式 (Predict_Swin_VIT_ResNet.py, Predict_ViT-H.py 等) 評估訓練完成的模型
