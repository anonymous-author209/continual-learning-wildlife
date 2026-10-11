import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoModelForImageClassification
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Define transforms
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ────────────────────────────────────────────────
# Path settings
# ────────────────────────────────────────────────
TEST_DATA_ROOT = "test"
MODEL_PATH = "/path/to/model/file"
# Load test dataset
test_dataset = datasets.ImageFolder(TEST_DATA_ROOT, transform=transform)

# Force class order to be '00' ~ '17'
class_names = [str(i).zfill(2) for i in range(18)]
test_dataset.classes = class_names
test_dataset.class_to_idx = {name: i for i, name in enumerate(class_names)}

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

species_names = [
    "Civet", "Emerald Dove", "Formosan Gem-faced Civet", "Formosan Ferret-badger",
    "Formosan Sambar Deer", "Formosan Serow", "Formosan Sika Deer",
    "Formosan Yellow-throated Marten", "Formosan Macaque", "Formosan Wild Boar",
    "Mouse", "Reeves's Muntjac", "Swinhoe's Pheasant", "Taiwan Partridge",
    "Urva", "Cat", "Dog", "Squirrel"
]

# Load model
print("📦 Loading model...")
model = AutoModelForImageClassification.from_pretrained(
    MODEL_PATH,
    num_labels=18,
    ignore_mismatched_sizes=True
)
model = model.to(device)
model.eval()

# ────────────────────────────────────────────────
# Evaluation loop
# ────────────────────────────────────────────────
all_preds, all_labels, misclassified = [], [], []
correct, total = 0, 0

with torch.no_grad():
    with torch.cuda.amp.autocast(): # Use 4090 FP16 acceleration
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="🔍 Evaluating")):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).logits
            predicted = torch.argmax(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            batch_preds = predicted.cpu().numpy()
            batch_labels = labels.cpu().numpy()
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)

            # Record paths to track errors
            start_idx = batch_idx * test_loader.batch_size
            batch_paths = [test_dataset.samples[i][0] for i in range(start_idx, min(start_idx + inputs.size(0), len(test_dataset)))]
            for i, (p, t, path) in enumerate(zip(batch_preds, batch_labels, batch_paths)):
                if p != t:
                    misclassified.append({
                        'file_path': path,
                        'true_species': species_names[t],
                        'predicted_species': species_names[p]
                    })

# ────────────────────────────────────────────────
# Metric calculation
# ────────────────────────────────────────────────
overall_accuracy = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds, labels=list(range(18)))
class_accuracies = cm.diagonal() / (cm.sum(axis=1) + 1e-12)

# F1 Score
beta = 1
p, r, f1, sup = precision_recall_fscore_support(
    all_labels, all_preds, labels=list(range(18)), beta=beta, zero_division=0
)

# Save CSV
report_data = {
    "Class": [f"{i:02d} ({species_names[i]})" for i in range(18)] + ["Overall Accuracy"],
    "Precision": list(p) + [np.mean(p)],
    "Recall": list(r) + [np.mean(r)],
    "F1 Score": list(f1) + [np.nan],
    "Accuracy": list(class_accuracies) + [overall_accuracy]
}
pd.DataFrame(report_data).to_csv("test_evaluation_with_f05.csv", index=False)
if misclassified:
    pd.DataFrame(misclassified).to_csv("misclassified_images.csv", index=False)

print(f"\n✅ Overall Test Accuracy: {overall_accuracy:.4f}")

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
print(f"\n✅ confusion_matrix_final_publication.png")
#plt.show()

# ====================== Per-class Average Precision ======================
print("\n" + "="*60)
print("Per-class Average Precision (Precision per class)")
print("="*60)

per_class_precision = []
supports = sup.tolist() if hasattr(sup, 'tolist') else sup

for i in range(18):
    prec = float(p[i])
    support = int(supports[i])
    class_name = species_names[i]
    
    if support > 0:
        print(f"Class {i:02d} ({class_name:25s}): Precision = {prec:.4f} (Support: {support})")
        per_class_precision.append(prec)
    else:
        print(f"Class {i:02d} ({class_name:25s}): No samples")
        per_class_precision.append(0.0)

# Macro Average Precision
macro_avg_precision = np.mean([p[i] for i in range(18) if supports[i] > 0]) if any(supports) else 0.0
print(f"\nMacro Average Precision: {macro_avg_precision:.4f}")

# Save CSV
ap_results = {
    "Class_ID": [f"{i:02d}" for i in range(18)],
    "Species": species_names,
    "Average_Precision": [float(p[i]) for i in range(18)],
    "Support": supports,
    "Per_class_Accuracy": class_accuracies.tolist() if hasattr(class_accuracies, 'tolist') else class_accuracies
}

ap_df = pd.DataFrame(ap_results)

# Add Macro row
ap_df.loc[len(ap_df)] = ["Macro", "Macro Average", macro_avg_precision, sum(supports), np.nan]
ap_df.to_csv("per_class_average_precision.csv", index=False)

print("\n✅ save  Average Precision : per_class_average_precision.csv")