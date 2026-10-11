import torch
import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader #, Dataset
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from transformers import ViTHybridForImageClassification

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ────────────────────────────────────────────────
# 1. Configuration and species alignment (based on your config.json)
# ────────────────────────────────────────────────
MODEL_PATH = "/path/to/model/file"
TEST_DIR = "test"

species_names = [
    "Civet", "Emerald_Dove", "Foemosan_Gem-faced_Civet", "Formosan_Ferret-badger",
    "Formosan_Sambar_Deer", "Formosan_Serow", "Formosan_Sika_Deer",
    "Formosan_Yellow-throated_Marten", "Formosan_macaque", "Formosan_wild_boar",
    "Mouse", "Reeves's_Muntjac", "Swinhoe's_Pheasant", "Taiwan_Partridge",
    "Urva", "cat", "dog", "squirrel"
]

# ────────────────────────────────────────────────
# 2. Model and data loading
# ────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_dataset.class_to_idx = {str(i).zfill(2): i for i in range(18)}
test_dataset.classes = [str(i).zfill(2) for i in range(18)]
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

print("📦 Loading ViT-Hybrid model weights...")
model = ViTHybridForImageClassification.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(device)
model.eval()

# ────────────────────────────────────────────────
# 3. Run inference
# ────────────────────────────────────────────────
all_preds, all_labels = [], []
with torch.no_grad():
    with torch.cuda.amp.autocast():
        for inputs, labels in tqdm(test_loader, desc="🔍 Inferring"):
            outputs = model(inputs.to(device)).logits
            all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            all_labels.extend(labels.numpy())

# ────────────────────────────────────────────────
# 4. Metric calculation and detailed output
# ────────────────────────────────────────────────
overall_acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds, labels=list(range(18)))
class_accuracies = cm.diagonal() / (cm.sum(axis=1) + 1e-12) # Avoid division by zero

print("\n" + "="*60)
print(f"📊 Final Test Report | Overall Accuracy: {overall_acc:.4f}")
print("="*60)
for i, (name, acc) in enumerate(zip(species_names, class_accuracies)):
    print(f"[{i:02d}] {name:<30} | Accuracy: {acc:.4f}")

# F1 Score
p, r, f1, sup = precision_recall_fscore_support(all_labels, all_preds, beta=1, labels=list(range(18)), zero_division=0)

# Save complete CSV report
report_df = pd.DataFrame({
    "Class_ID": [f"{i:02d}" for i in range(18)],
    "Species": species_names,
    "Accuracy": class_accuracies,
    "Precision": p,
    "Recall": r,
    "F1_Score": f1,
    "Support": sup
})
report_df.to_csv("test_full_evaluation.csv", index=False)

# ────────────────────────────────────────────────
# 5. Journal-level confusion matrix (extreme large font + optimized colorbar version)
# ────────────────────────────────────────────────
LABEL_FONT_SIZE = 30  # Species names & axis labels
ANNOT_FONT_SIZE = 25  # Numbers inside the matrix
CBAR_TICK_SIZE = 30   # Colorbar tick value size

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.weight': 'bold',
    'figure.dpi': 300
})

fig, ax = plt.subplots(figsize=(24, 20))

# Set colorbar via cbar_kws
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Blues",
    xticklabels=species_names, 
    yticklabels=species_names,
    annot_kws={"size": ANNOT_FONT_SIZE, "weight": "bold"},
    cbar_kws={"shrink": 0.8, "label": "Number of Predictions"},
    ax=ax
)

# ─── Key setting: Adjust the font size of the right colorbar ───
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE) 
cbar.set_label("Number of Predictions", size=LABEL_FONT_SIZE, weight='bold')

# Set axis species text
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=LABEL_FONT_SIZE, fontweight='bold')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=LABEL_FONT_SIZE, fontweight='bold')

# Set axis titles
ax.set_xlabel("Predicted Species", fontsize=LABEL_FONT_SIZE + 8, fontweight='bold', labelpad=20)
ax.set_ylabel("True Species", fontsize=LABEL_FONT_SIZE + 8, fontweight='bold', labelpad=20)

plt.tight_layout()

# Save file
plt.savefig("confusion_matrix.png", bbox_inches="tight")
print(f"\n✅ save confusion_matrix_final_publication.png")
#plt.show()