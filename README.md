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

---

## File Structure
```text
continual-learning-wildlife/
├── Continual learning/                     # Source code 
│   ├── Predict_Swin_VIT_ResNet.py          # Swin, ViT, ResNet prediction script
│   ├── Predict_ViT-H.py                    # ViT-H prediction script
│   ├── Train Resnet, Swin, ViT/
│   │   ├── Finetune(CrossEntropy).py       # Main training script
│   │   ├── EWC.py                          # Main training script
│   │   ├── VCL.py                          # Main training script
│   │   └── FIP.py                          # Main training script
│   └── Train ViT-H/
│       ├── Finetune(CrossEntropy).py       # Main training script
│       ├── EWC.py                          # Main training script
│       ├── VCL.py                          # Main training script
│       └── FIP.py                          # Main training script
│
├── Transfer learning/                      # Source code
│   ├── Predict_Resnet_Swin_ViT.py          # ResNet, Swin, ViT prediction script
│   ├── Predict_ViT-H.py                    # ViT-H prediction script
│   ├── ResNet-34/
│   │   └── train.py                        # Main training script
│   ├── Swin Transformer/
│   │   └── train.py                        # Main training script
│   ├── ViT/
│   │   └── train.py                        # Main training script
│   └── ViT-H/
│       └── train.py                        # Main training script
│
├── requirements.txt                        # Python dependencies
├── README.md                               # Project documentation
└── LICENSE                                 # License file
```

---

## 🔄 Workflow

The overall workflow of this repository follows three main stages:

1. Data Preparation
  * Please download the task-specific dataset from Hugging Face: [[Transfer Learning](https://anonymous-hf.com/a/gc232h784xp3/)][[Continual Learning](https://anonymous-hf.com/a/n946ij7q1wva/)] and place the dataset in the same directory as the training scripts.
  * Experimental result weights can be found [[Transfer Learning](https://anonymous-hf.com/a/c3imga7exsy0/)][[Continual Learning(Region)](https://anonymous-hf.com/a/qgkxbqqn3qck/)][[Continual Learning(Taxonomic)](https://anonymous-hf.com/a/7az71pmleqvm/)].

2. Model Training
  * For Continual Learning, use scripts under Continual learning/Train Resnet, Swin, ViT/ and Train ViT-H/ (e.g., EWC.py, VCL.py, FIP.py).
  * For Transfer Learning, use scripts under Transfer learning/ResNet-34/, Swin Transformer/, ViT/, and ViT-H/.

3. Model Evaluation
  * Run prediction scripts (Predict_Swin_VIT_ResNet.py, Predict_ViT-H.py, etc.) to evaluate trained models.
