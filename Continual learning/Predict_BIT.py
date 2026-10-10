import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTHybridForImageClassification
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
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
print(f"Using device: {device}")

# Define transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load test dataset
test_dataset = datasets.ImageFolder('test_final', transform=transform)

# Define correct class order (numerical, zero-padded)
class_names = [str(i).zfill(2) for i in range(18)]  # ['00', '01', ..., '17']
class_to_idx = {name: i for i, name in enumerate(class_names)}
test_dataset.class_to_idx = class_to_idx
test_dataset.classes = class_names

print(f"Test dataset classes: {test_dataset.classes}")
print(f"Test dataset class_to_idx: {test_dataset.class_to_idx}")

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Test dataset sample count: {len(test_dataset)}")
print(f"Test dataset number of classes: {len(test_dataset.classes)}")
print(f"Test label distribution: {dict(sorted({label: count for label, count in zip(*np.unique(test_dataset.targets, return_counts=True))}.items()))}")

# Verify label range
if min(test_dataset.targets) < 0 or max(test_dataset.targets) > 17:
    print("Warning: Test dataset contains invalid labels outside 0-17 range!")
    exit(1)

# Species names (00 → 17 的對應名稱)
species_names = [
    "Cat",
    "Civet",
    "Dog",
    "Emerald Dove",
    "Formosan Gem-faced Civet",
    "Formosan Ferret-badger",
    "Formosan Macaque",
    "Formosan Sambar Deer",
    "Formosan Serow",
    "Formosan Sika Deer",
    "Formosan Wild Boar",
    "Formosan Yellow-throated Marten",
    "Mouse",
    "Reeves's Muntjac",
    "Squirrel",
    "Swinhoe's Pheasant",
    "Taiwan Partridge",
    "Urva"
]

# Load the best model
model_path = "./model"
model = ViTHybridForImageClassification.from_pretrained(
    model_path,
    # 不加 num_labels 和 ignore_mismatched_sizes，避免觸發 mask_token bug
)
model = model.to(device)
model.eval()

# Debug: 確認載入的類別數
print(f"Loaded model num_labels: {model.num_labels}")
print(f"Classifier output features: {model.classifier.out_features}")  # 應該是 18

# Initialize lists
all_preds = []
all_labels = []
misclassified = []

# Evaluate model
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs).logits
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        start_idx = batch_idx * test_loader.batch_size
        batch_paths = [test_dataset.samples[i][0] for i in range(start_idx, min(start_idx + inputs.size(0), len(test_dataset)))]

        batch_preds = predicted.cpu().numpy()
        batch_labels = labels.cpu().numpy()
        all_preds.extend(batch_preds)
        all_labels.extend(batch_labels)

        for i, (pred, true, path) in enumerate(zip(batch_preds, batch_labels, batch_paths)):
            if pred != true:
                misclassified.append({
                    'file_path': path,
                    'true_label': true,
                    'predicted_label': pred
                })

# Overall accuracy
overall_accuracy = accuracy_score(all_labels, all_preds)
print(f"\nOverall Test Accuracy: {overall_accuracy:.4f} ({correct}/{total})")

# Per-class accuracy
class_correct = [0] * 18
class_total = [0] * 18
for pred, label in zip(all_preds, all_labels):
    class_total[label] += 1
    if pred == label:
        class_correct[label] += 1

print("\nPer-class Accuracy:")
class_accuracies = {}
for i in range(18):
    if class_total[i] > 0:
        acc = class_correct[i] / class_total[i]
        class_accuracies[f"Class {i}"] = acc
        print(f"Class {i:02d}: {acc:.4f} ({class_correct[i]}/{class_total[i]})")
    else:
        class_accuracies[f"Class {i}"] = 0.0
        print(f"Class {i:02d}: No samples")

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds, labels=list(range(18)))

# F1 Score
beta = 1
precisions, recalls, f1_scores, supports = precision_recall_fscore_support(
    all_labels, all_preds, labels=list(range(18)), beta=beta, zero_division=0
)

macro_precision = np.mean(precisions)
macro_recall = np.mean(recalls)
macro_f1 = ((1 + beta**2) * macro_precision * macro_recall) / (beta**2 * macro_precision + macro_recall + 1e-12)

weighted_precision = np.average(precisions, weights=supports)
weighted_recall = np.average(recalls, weights=supports)
weighted_f1 = ((1 + beta**2) * weighted_precision * weighted_recall) / (beta**2 * weighted_precision + weighted_recall + 1e-12)

print("\n" + "="*60)
print("Per-class F1 Score")
print("="*60)
for i in range(18):
    p, r, f1, sup = precisions[i], recalls[i], f1_scores[i], supports[i]
    if sup > 0:
        print(f"Class {i:2d}: P={p:.4f}, R={r:.4f}, F1={f1:.4f}, Support={sup}")
    else:
        print(f"Class {i:2d}: No samples")

print(f"\nMacro Avg    -> P: {macro_precision:.4f}, R: {macro_recall:.4f}, F1: {macro_f1:.4f}")
print(f"Weighted Avg -> P: {weighted_precision:.4f}, R: {weighted_recall:.4f}, F1: {weighted_f1:.4f}")

# ────────────────────────────────────────────────
# Save results to CSV (with species names)
# ────────────────────────────────────────────────

f1_results = {
    "Class": [f"{i:02d} ({species_names[i]})" for i in range(18)] + ["Macro Avg", "Weighted Avg", "Overall Accuracy"],
    "Precision": list(precisions) + [macro_precision, weighted_precision, np.nan],
    "Recall": list(recalls) + [macro_recall, weighted_recall, np.nan],
    "F1 Score": list(f1_scores) + [macro_f1, weighted_f1, np.nan],
    "Support": list(supports) + [np.sum(supports), np.sum(supports), len(all_labels)],
    "Accuracy": [class_accuracies.get(f"Class {i}", 0.0) for i in range(18)] + [np.nan, np.nan, overall_accuracy]
}
f1_df = pd.DataFrame(f1_results)
f1_df.to_csv("test_evaluation.csv", index=False)
print("\nDetailed results with F1 saved to test_evaluation_with_f1.csv")

results = {
    "Class": [f"{i:02d} ({species_names[i]})" for i in range(18)] + ["Overall"],
    "Accuracy": [class_accuracies.get(f"Class {i}", 0.0) for i in range(18)] + [overall_accuracy]
}
df = pd.DataFrame(results)
#df.to_csv("test_evaluation_results.csv", index=False)
#print("Evaluation results saved to test_evaluation_results.csv")

# Misclassified
if misclassified:
    misclassified_df = pd.DataFrame(misclassified)
    misclassified_df.to_csv("misclassified_images.csv", index=False)
    print("\nMisclassified images saved to misclassified_images.csv")
    print("\nMisclassified Images (showing first 10):")
    for item in misclassified[:10]:
        tl = species_names[item['true_label']]
        pl = species_names[item['predicted_label']]
        print(f"Image: {item['file_path']}, True: {item['true_label']} ({tl}), Pred: {item['predicted_label']} ({pl})")
    if len(misclassified) > 10:
        print(f"... and {len(misclassified)-10} more.")
else:
    print("\nNo misclassified images found.")

# ────────────────────────────────────────────────
# 5. 期刊級混淆矩陣 (極致大字體 + 圖例優化版)
# ────────────────────────────────────────────────
LABEL_FONT_SIZE = 30  # 物種名稱 & 座標軸標籤
ANNOT_FONT_SIZE = 25  # 矩陣內部數字
CBAR_TICK_SIZE = 30   # 右側圖例數值大小

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.weight': 'bold',
    'figure.dpi': 300
})

fig, ax = plt.subplots(figsize=(24, 20))

# 透過 cbar_kws 設定圖例
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

# ─── 關鍵設定：調整右側圖例 (Colorbar) 的字體大小 ───
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE) 
cbar.set_label("Number of Predictions", size=LABEL_FONT_SIZE, weight='bold')

# 設定座標軸物種文字
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=LABEL_FONT_SIZE, fontweight='bold')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=LABEL_FONT_SIZE, fontweight='bold')

# 設定座標軸標題
ax.set_xlabel("Predicted Species", fontsize=LABEL_FONT_SIZE + 8, fontweight='bold', labelpad=20)
ax.set_ylabel("True Species", fontsize=LABEL_FONT_SIZE + 8, fontweight='bold', labelpad=20)

plt.tight_layout()

# 存檔
plt.savefig("ViT-H_confusion_matrix.png", bbox_inches="tight")
print(f"\n✅ save confusion_matrix_final_publication.png")
#plt.show()
# ====================== 新增：每個類別的 Average Precision ======================
print("\n" + "="*60)
print("Per-class Average Precision (Precision per class)")
print("="*60)

per_class_precision = []
for i in range(18):
    p = precisions[i]
    sup = supports[i]
    class_name = species_names[i]
    if sup > 0:
        print(f"Class {i:02d} ({class_name:25s}): Precision = {p:.4f}  (Support: {sup})")
        per_class_precision.append(p)
    else:
        print(f"Class {i:02d} ({class_name:25s}): No samples")
        per_class_precision.append(0.0)

# 計算整體平均 Precision (Macro Average Precision)
macro_avg_precision = np.mean([p for p in precisions if supports[list(precisions).index(p)] > 0] or [0])

print(f"\nMacro Average Precision: {macro_avg_precision:.4f}")

# 儲存每個類別的 Average Precision 到獨立 CSV
ap_results = {
    "Class_ID": [f"{i:02d}" for i in range(18)],
    "Species": species_names,
    "Average_Precision": [float(p) for p in precisions],
    "Support": supports.tolist(),
    "Per_class_Accuracy": [class_accuracies.get(f"Class {i}", 0.0) for i in range(18)]
}

ap_df = pd.DataFrame(ap_results)

# 加入 Macro Average 列
ap_df.loc[len(ap_df)] = ["Macro", "Macro Average", macro_avg_precision, np.sum(supports), np.nan]

#ap_df.to_csv("per_class_average_precision.csv", index=False)
#print("\n✅ 每個類別的 Average Precision 已儲存至: per_class_average_precision.csv")