# ==============================================================================
# ------------------------------- ResNet -------------------------------------
# ==============================================================================


# ==============================================================================
# -- Imports -------------------------------------------------------------------
# ==============================================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 必須在 pyplot 之前設定
import os

import seaborn as sns
import datetime
from PIL import Image, ImageFile
import wandb    #CMD wandb login  # 輸入你的 API key

from torchvision.datasets import ImageFolder
from torchvision.transforms import Normalize, Resize, ToTensor, Compose
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter
import torchvision.transforms as transforms

from transformers import TrainingArguments, Trainer
from transformers import AutoImageProcessor, ResNetForImageClassification

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# 忽略圖片損壞錯誤
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 定義安全讀圖 Dataset 類別
class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except (OSError, RuntimeError, ValueError) as e:
            print(f"❗ Skipping broken image at index {index}: {e}")
            return self.__getitem__((index + 1) % len(self))
        
# 生成時間戳記
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

# ✅ 初始化 wandb（每次超參數搜尋會重啟）
wandb.init(
  project="fine-tuning",     # ✅ 可以共用同一 project
  name="resnet-34-384-mis",  # ✅ 改這個來區分不同模型
  group="resnet",      # ✅ 可加 group 方便比較
  reinit=True
)
# ==============================================================================
# -- Check GPU Availability ----------------------------------------------------
# ==============================================================================

# # 檢查 GPU 是否可用，若不可用則拋出錯誤
# if not torch.cuda.is_available():
#     raise RuntimeError("GPU 不可用，請檢查 CUDA 是否正確安裝！")

# device = torch.device("cuda")  # 強制使用 GPU


# ==============================================================================
# -- Dataset Setup -------------------------------------------------------------
# ==============================================================================

dataset_path = "/path/to/dataset"   #origin dataset_path = "/mnt/g/animal-12K" /media/username/F2/

# G:\roboflow_ori\4.8K_vit
# G:\roboflow_ori\7.2K

# ==============================================================================
# -- Image Preprocessing -------------------------------------------------------
# ==============================================================================

# 設定影像處理

# google/vit-large-patch16-224  Imagenet-1K 
# google/vit-base-patch16-224-in21k Imagenet-21K

# 目前已使用：
#   B16-224、L16-224、L16-384
#   18、34、50、152

model_name = "microsoft/resnet-34"
local_config_path = "/path/to/resnet-34" 
#   #1)用於本地運行 (if need)

# resnet 18 snapshots/65a5785d9156231087c481e0c7dd33a5ff6f7e3e
# restnet 34    4d180fbdc81b4d5d4eaa86ab89bdd669200dd477
# restnet 50    34c2154c194f829b11125337b98c8f5f9965ff19
# restnet 152   189229b1590873cc46c03813abc7d0a7cb8f02b2
processor = AutoImageProcessor.from_pretrained(local_config_path) #model_name   原為：ViTImageProcessor、AutoFeatureExtractor
mu, sigma = processor.image_mean, processor.image_std
size = processor.size
# processor.size = {"height": 384, "width": 384}

# 設定轉換
transform = Compose([
    Resize((384, 384)),  # ResNet 輸入大小    
    # RandomCrop(224, padding=20),  # ✅ 只做小幅隨機裁剪，避免丟失目標
    # CenterCrop(224),  # ✅ 保持目標完整
    # RandomHorizontalFlip(p=0.5),  # ✅ 左右翻轉
    # RandomRotation(degrees=5),  # ✅ 限制旋轉範圍，防止過度變形
    # ColorJitter(brightness=0.05, contrast=0.05),  # ✅ 避免過度改變光照
    ToTensor(),
    Normalize(mean=mu, std=sigma)
])

# 設定顯示用的轉換（不含標準化）
display_transform = Compose([
    Resize((384, 384)),
    ToTensor()
])

# 載入自定義數據集
dataset = ImageFolder(root=dataset_path, transform=transform)
# 建立用於顯示的數據集（不含標準化）
display_dataset = ImageFolder(root=dataset_path, transform=display_transform)

# 建立標籤對應
itos = {i: label for i, label in enumerate(dataset.classes)}
stoi = {label: i for i, label in itos.items()}

# 切分訓練集 & 測試集
train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
trainds = torch.utils.data.Subset(dataset, train_indices)
valds = torch.utils.data.Subset(dataset, val_indices)
# 用於顯示的驗證集
display_valds = torch.utils.data.Subset(display_dataset, val_indices)

def collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])  # 圖片
    labels = torch.tensor([item[1] for item in batch])  # 標籤
    return {"pixel_values": pixel_values, "labels": labels}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# -- Model Setup ---------------------------------------------------------------
# ==============================================================================

model = ResNetForImageClassification.from_pretrained(  # ViTHybridForImageClassification，原為：ViTForImageClassification
    local_config_path,
    num_labels=len(itos),
    id2label=itos,
    label2id=stoi,
    ignore_mismatched_sizes=True
).to(device)  # 確保模型移動到 GPU

# # ✅ 凍結 ResNet-50 的前幾層（僅訓練最後幾層）
# for param in model.base_model.parameters():
#     param.requires_grad = False  # ✅ 凍結大部分層，不更新權重

# # ✅ 只解凍分類層（最後幾層）
# for param in model.classifier.parameters():
#     param.requires_grad = True

# ==============================================================================
# -- Training Configuration ----------------------------------------------------
# ==============================================================================

args = TrainingArguments(
    output_dir="./results-{model_name}",  # ✅ 改到外接磁碟
    save_strategy="epoch",
    evaluation_strategy="epoch",  #evaluation_strategy HF(v 4.40.2, New version :save_strategy)
    logging_strategy="epoch",  # 確保紀錄 train_loss
    learning_rate=2e-5, # resnet-降低學習率 (5e-6)、使用 cosine 調度器  原為：learning_rate=2e-5
    # lr_scheduler_type="cosine",  # resnet-使用 Cosine 調度器
    # warmup_ratio=0.1,  #  讓學習率在前 10% 的步驟內緩慢上升
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir='logs',
    fp16=torch.cuda.is_available(),  # 啟用半精度訓練（如果有 GPU）
    
    report_to="wandb",   # ✅ 用 wandb 紀錄
    run_name=f"fine-tune-{model_name}",
)


# 設置 Trainer
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=trainds,
    eval_dataset=valds,
    data_collator=collate_fn,
    # processing_class=processor,   HF(v 4.40.2 no use)
    compute_metrics=compute_metrics,
)


# ==============================================================================
# -- Training & Evaluation -----------------------------------------------------
# ==============================================================================

train_result = trainer.train()
trainer.save_model(f"fine-tuned-animal-18K-{model_name}")  # 儲存模型
trainer.evaluate()


# ==============================================================================
# -- Logging & Visualization ---------------------------------------------------
# ==============================================================================

# 提取 loss 和準確率記錄
# 在 Hugging Face Trainer 中TrainOutput 物件不包含 training_loss_history 屬性。
train_steps = [log["epoch"] for log in trainer.state.log_history if "loss" in log]
eval_steps = [log["epoch"] for log in trainer.state.log_history if "eval_loss" in log]     

loss_values = [log["loss"] for log in trainer.state.log_history if "loss" in log]
eval_loss_values = [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log]
eval_accuracy_values = [log["eval_accuracy"] for log in trainer.state.log_history if "eval_accuracy" in log]

# 附加時間戳記與模型名稱到輸出檔案名
safe_model_name = model_name.replace("/", "-")  #避免 FileNotFoundError: [Errno 2] No such file or directory
loss_curve_path = f"loss_curve_{safe_model_name}_{timestamp}.png"
accuracy_curve_path = f"accuracy_curve_{safe_model_name}_{timestamp}.png"
f1_scores_path = f"f1_scores_{safe_model_name}_{timestamp}.png"

def plot_loss_curve(train_steps, loss_values, eval_steps, eval_loss_values, save_path=loss_curve_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_steps, loss_values, label='Training Loss', linestyle='-', color='blue', marker='o')
    plt.plot(eval_steps, eval_loss_values, label='Validation Loss', linestyle='--', color='red', marker='s')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.savefig(save_path)
    print(f"Loss curve saved to {save_path}")

def plot_accuracy_curve(eval_steps, eval_accuracy_values, save_path=accuracy_curve_path):
    plt.figure(figsize=(10, 5))
    plt.plot(eval_steps, eval_accuracy_values, label='Validation Accuracy', linestyle='--', color='red', marker='s')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Curve")
    plt.legend()
    plt.savefig(save_path)
    print(f"Accuracy curve saved to {save_path}")

def plot_f1_score_curve(f1_scores, save_path=f1_scores_path):
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(f1_scores)), list(f1_scores.values()), tick_label=list(f1_scores.keys()))
    plt.xlabel("Classes")
    plt.ylabel("F1-score")
    plt.title("F1-score per Class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"F1-score curve saved to {save_path}")

plot_loss_curve(train_steps, loss_values, eval_steps, eval_loss_values)
plot_accuracy_curve(eval_steps, eval_accuracy_values)

# 評估模型
outputs = trainer.predict(valds)
y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)

# ==============================================================================
# -- confusion_matrix ---------------------------------------------------------
# ==============================================================================

# 繪製混淆矩陣  (cm)
# cm = confusion_matrix(y_true, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(itos.values()))
# disp.plot(xticks_rotation=45)
# plt.show()

# 繪製混淆矩陣  (Normalized cm)
def plot_normalized_confusion_matrix(y_true, y_pred, save_path=f'normalized_confusion_matrix_{safe_model_name}_{timestamp}.png'):
    cm = confusion_matrix(y_true, y_pred, normalize='true')  #  標準化到 0~1
    
    plt.figure(figsize=(16, 8))
    sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=list(itos.values()),
            yticklabels=list(itos.values())
    )
    # 加上標籤與標題
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f'Normalized Confusion Matrix_{safe_model_name}')
    # 自動調整 layout
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # 儲存圖像
    plt.savefig(save_path, dpi=300)
    plt.close()  # 防止記憶體累積或圖未釋放
    print(f"Normalized Confusion Matrix saved to {save_path}")

# 計算準確率
accuracy = accuracy_score(y_true, y_pred)
print(f"模型在驗證集上的準確率: {accuracy:.4f}")
plot_normalized_confusion_matrix(y_true, y_pred)


# ==============================================================================
# -- Classification Report & GenerateF1-score Plot-----------------------------
# ==============================================================================

report = classification_report(y_true, y_pred, target_names=list(itos.values()), output_dict=True)
report_path = f"classification_report_{safe_model_name}_{timestamp}.txt"
with open(report_path, "w") as f:
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            f.write(f"{label}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1-score={metrics['f1-score']:.4f}\n")

print(f"Classification report saved to {report_path}")

# 取得 F1-score 並繪製圖表
f1_scores = {label: metrics["f1-score"] for label, metrics in report.items() if isinstance(metrics, dict)}
plot_f1_score_curve(f1_scores)


# ==============================================================================
# -- Error Analysis - Display Misclassified Images ----------------------------
# ==============================================================================

def show_misclassified_images(display_dataset, val_indices, y_true, y_pred, itos, 
                            max_images=20, save_dir="misclassified_images"):
    """
    顯示並保存辨識錯誤的圖片
    
    Args:
        display_dataset: 用於顯示的數據集（未標準化）
        val_indices: 驗證集索引
        y_true: 真實標籤
        y_pred: 預測標籤
        itos: 索引到標籤的映射
        max_images: 最多顯示的圖片數量
        save_dir: 保存錯誤圖片的目錄
    """
    
    # 找出錯誤預測的索引
    misclassified_indices = np.where(y_true != y_pred)[0]
    
    if len(misclassified_indices) == 0:
        print("🎉 所有圖片都被正確分類了！")
        return
    
    print(f"❌ 總共發現 {len(misclassified_indices)} 張錯誤分類的圖片")
    
    # 創建保存目錄
    os.makedirs(save_dir, exist_ok=True)
    
    # 限制顯示數量
    show_indices = misclassified_indices[:max_images]
    
    # 計算子圖佈局
    n_images = len(show_indices)
    n_cols = 4  # 每行4張圖
    n_rows = (n_images + n_cols - 1) // n_cols
    
    plt.figure(figsize=(20, 5 * n_rows))
    
    for i, idx in enumerate(show_indices):
        # 取得原始驗證集中的索引
        original_idx = val_indices[idx]
        image, _ = display_dataset[original_idx]
        
        # 轉換張量為可顯示的格式
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
            image = np.clip(image, 0, 1)
        
        true_label = itos[y_true[idx]]
        pred_label = itos[y_pred[idx]]
        
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(image)
        plt.title(f"True: {true_label}\nPred: {pred_label}", fontsize=10, color='red')
        plt.axis('off')
        
        # 保存個別圖片
        save_path = os.path.join(save_dir, f"error_{i+1:03d}_true_{true_label}_pred_{pred_label}.png")
        plt.imsave(save_path, image)
    
    plt.tight_layout()
    overview_path = os.path.join(save_dir, f"misclassified_overview_{safe_model_name}_{timestamp}.png")
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ 錯誤分類圖片概覽已保存至: {overview_path}")
    print(f"✅ 個別圖片已保存至目錄: {save_dir}")
    
    # 統計每個類別的錯誤數量
    error_stats = {}
    for idx in misclassified_indices:
        true_label = itos[y_true[idx]]
        pred_label = itos[y_pred[idx]]
        
        if true_label not in error_stats:
            error_stats[true_label] = {}
        if pred_label not in error_stats[true_label]:
            error_stats[true_label][pred_label] = 0
        error_stats[true_label][pred_label] += 1
    
    print("\n📊 錯誤分類統計:")
    for true_label, pred_dict in error_stats.items():
        print(f"\n真實標籤: {true_label}")
        for pred_label, count in pred_dict.items():
            print(f"  → 被誤認為 {pred_label}: {count} 次")

def analyze_prediction_confidence(outputs, y_true, y_pred, itos, threshold=0.5):
    """
    分析預測信心度
    """
    predictions = outputs.predictions
    probabilities = torch.softmax(torch.tensor(predictions), dim=1).numpy()
    
    # 取得預測的最高機率
    max_probs = np.max(probabilities, axis=1)
    
    # 找出低信心度的預測
    low_confidence_mask = max_probs < threshold
    misclassified_mask = y_true != y_pred
    
    print(f"\n🎯 預測信心度分析:")
    print(f"總樣本數: {len(y_true)}")
    print(f"低信心度預測 (<{threshold}): {np.sum(low_confidence_mask)} 個")
    print(f"錯誤預測: {np.sum(misclassified_mask)} 個")
    print(f"低信心度且錯誤的預測: {np.sum(low_confidence_mask & misclassified_mask)} 個")
    
    # 繪製信心度分佈圖
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(max_probs[~misclassified_mask], bins=50, alpha=0.7, label='Correct prediction', color='green')
    plt.hist(max_probs[misclassified_mask], bins=50, alpha=0.7, label='Wrong prediction', color='red')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Count')
    plt.title('Prediction Confidence Distribution')
    plt.legend()
    plt.axvline(x=threshold, color='black', linestyle='--', label=f'Threshold {threshold}')
    
    # Correct prediction    正確預測
    # Wrong prediction      錯誤預測
    # Prediction Confidence 預測信心度
    # Count                 數量
    # Prediction Confidence Distribution    預測信心度分佈
    # Threshold             閾值

    # 預測信心度    Prediction Confidence
    # 錯誤分類  Misclassification
    # 真實標籤  Ground Truth Label
    # 預測標籤  Predicted Label
    # 信心度分佈    Confidence Distribution
    # 正確預測  Correct Predictions
    # 錯誤預測  Incorrect Predictions
    # 低信心度  Low Confidence
    # 閾值  Threshold
    # 各類別平均信心度  Average Confidence per Class


    plt.subplot(1, 2, 2)
    # 計算每個類別的平均信心度
    class_confidence = {}
    for i, class_name in itos.items():
        mask = y_pred == i
        if np.sum(mask) > 0:
            class_confidence[class_name] = np.mean(max_probs[mask])
    
    classes = list(class_confidence.keys())
    confidences = list(class_confidence.values())
    
    plt.bar(range(len(classes)), confidences)
    plt.xlabel('Class')
    plt.ylabel('Average Confidence')
    plt.title('Average Confidence per Class')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    
    plt.tight_layout()
    confidence_path = f"prediction_confidence_{safe_model_name}_{timestamp}.png"
    plt.savefig(confidence_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ 預測信心度分析圖已保存至: {confidence_path}")

# 執行錯誤分析
print("\n" + "="*80)
print("開始錯誤分析...")
print("="*80)

show_misclassified_images(display_dataset, val_indices, y_true, y_pred, itos, 
                         max_images=20, save_dir=f"misclassified_{safe_model_name}_{timestamp}")

analyze_prediction_confidence(outputs, y_true, y_pred, itos, threshold=0.8)