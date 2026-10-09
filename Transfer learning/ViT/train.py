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
# from ray import tune
import wandb    #CMD wandb login  # 輸入你的 API key

from torchvision.datasets import ImageFolder
from torchvision.transforms import Normalize, Resize, ToTensor, Compose
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter
import torchvision.transforms as transforms

from transformers import TrainingArguments, Trainer
from transformers import ViTImageProcessor, ViTForImageClassification

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
  name="vit-large-384-mis",  # ✅ 改這個來區分不同模型
  group="vit",      # ✅ 可加 group 方便比較
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
# animal-sum
# animal-sum-18K

# ==============================================================================
# -- Image Preprocessing -------------------------------------------------------
# ==============================================================================

# 設定影像處理

# google/vit-large-patch16-224  Imagenet-1K 
# google/vit-base-patch16-224-in21k Imagenet-21K

# 目前已使用：
#   B16-224、L16-224、L16-384

# B16：/home/raychag0122/Real_project/NTUT_repo/LLM/Vit/Fine-tuning/Hugging_face/google/vit-base-patch16-224/preprocessor_config.json
# 原為 model_name，若出現 [OSError: google/vit-base-patch16-224 does not appear to have a file named preprocessor_config.json.
#  Checkout 'https://huggingface.co/google/vit-base-patch16-224/tree/main' for available files.]
# 1)可改為local_config_path以本地運行，注：需事先將model.safetensors、preprocessor_config.json、pytorch_model.bin 等下載到相應的位置
# 2)也可改為以 huggingface-cli download google/vit-hybrid-base-bit-384 下載
# ls /home/raychag0122/.cache/huggingface/hub/models--google--vit-hybrid-base-bit-384/snapshots/e8cc9d5fe904240b3b0035837daeabff06202bf2    (存於.cache下)

model_name = "google/vit-large-patch32-384"
local_config_path = "/path/to/ViT" 
#   #1)用於本地運行 (if need)
# google/vit-base-patch16-22   /New/3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3
# google/vit-large-patch16-224  /9e2727f4250d3973839eecfa5c4b42e41b709a50
# google/vit-hybrid-base-bit-384    /e8cc9d5fe904240b3b0035837daeabff06202bf2

# 4090-set

# google/vit-base-patch16-22   /New/3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3
# google/vit-large-patch16-22   /9e2727f4250d3973839eecfa5c4b42e41b709a50

# google/vit-base-patch32-384   /path/d285164d04ed60526ea1b462ba8cc7afbe53a31b
# google/vit-large-patch32-384   /path/a2b30ad36d02e99f045cd2ecfc71e0ae16991efa

# google/vit-hybrid-base-bit-384    /
#   path    /home/username/.cache/huggingface/hub/models--google--vit-hybrid-base-bit-384/snapshots/e8cc9d5fe904240b3b0035837daeabff06202bf2

processor = ViTImageProcessor.from_pretrained(local_config_path) #model_name   原為：ViTImageProcessor、AutoFeatureExtractor
mu, sigma = processor.image_mean, processor.image_std
size = processor.size

# 設定轉換
transform = transforms.Compose([
    transforms.Resize((size['height'], size['width'])),
    # transforms.RandomHorizontalFlip(p=0.5),  # 左右翻轉
    # transforms.RandomRotation(degrees=5),  # 隨機旋轉
    # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 調整亮度/對比度
    transforms.ToTensor(),
    transforms.Normalize(mean=mu, std=sigma)
])

# 設定顯示用的轉換（不含標準化）
display_transform = Compose([
    Resize((384, 384)),
    ToTensor()
])

# 載入自定義數據集
dataset = SafeImageFolder(root=dataset_path, transform=transform)
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

model = ViTForImageClassification.from_pretrained(
    local_config_path,
    num_labels=len(itos),
    id2label=itos,
    label2id=stoi,
    ignore_mismatched_sizes=True
).to(device)


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
    output_dir="./results{model_name}",  # ✅ 改到外接磁碟
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

def save_misclassification_report(dataset, display_dataset, val_indices, y_true, y_pred, itos, 
                                outputs, save_dir="misclassified_images"):
    """
    將所有錯誤分類結果保存為詳細的txt報告
    
    Args:
        dataset: 原始數據集（用於獲取路徑）
        display_dataset: 用於顯示的數據集
        val_indices: 驗證集索引
        y_true: 真實標籤
        y_pred: 預測標籤
        itos: 索引到標籤的映射
        outputs: 模型輸出結果（包含預測概率）
        save_dir: 保存目錄
    """
    
    # 找出錯誤預測的索引
    misclassified_indices = np.where(y_true != y_pred)[0]
    
    if len(misclassified_indices) == 0:
        print("🎉 所有圖片都被正確分類了！")
        return
    
    # 創建保存目錄
    os.makedirs(save_dir, exist_ok=True)
    
    # 獲取預測概率
    predictions = outputs.predictions
    probabilities = torch.softmax(torch.tensor(predictions), dim=1).numpy()
    
    # 生成報告文件名
    report_path = os.path.join(save_dir, f"misclassification_report_{safe_model_name}_{timestamp}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # 寫入標題和概要
        f.write("="*80 + "\n")
        f.write("錯誤分類詳細報告\n")
        f.write("="*80 + "\n")
        f.write(f"模型: {model_name}\n")
        f.write(f"生成時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"總驗證樣本數: {len(y_true)}\n")
        f.write(f"錯誤分類數量: {len(misclassified_indices)}\n")
        f.write(f"錯誤率: {len(misclassified_indices)/len(y_true)*100:.2f}%\n")
        f.write("\n" + "="*80 + "\n")
        f.write("詳細錯誤列表\n")
        f.write("="*80 + "\n\n")
        
        # 逐一記錄每個錯誤
        for i, idx in enumerate(misclassified_indices):
            original_idx = val_indices[idx]
            true_label = itos[y_true[idx]]
            pred_label = itos[y_pred[idx]]
            
            # 獲取該樣本的所有類別概率
            sample_probs = probabilities[idx]
            max_prob = np.max(sample_probs)
            
            # 獲取top-3預測
            top3_indices = np.argsort(sample_probs)[-3:][::-1]
            top3_labels = [itos[j] for j in top3_indices]
            top3_probs = [sample_probs[j] for j in top3_indices]
            
            f.write(f"錯誤 #{i+1:03d}\n")
            f.write(f"  原始索引: {original_idx}\n")
            f.write(f"  真實標籤: {true_label}\n")
            f.write(f"  預測標籤: {pred_label}\n")
            f.write(f"  預測信心度: {max_prob:.4f}\n")
            f.write(f"  Top-3 預測:\n")
            for j, (label, prob) in enumerate(zip(top3_labels, top3_probs)):
                marker = "★" if label == pred_label else " "
                f.write(f"    {j+1}. {marker} {label}: {prob:.4f}\n")
            # 獲取圖片路徑
            try:
                image_path = dataset.samples[original_idx][0]
            except (AttributeError, IndexError):
                image_path = f"Index: {original_idx} (路徑獲取失敗)"
            
            f.write(f"  圖片路徑: {image_path}\n")
            f.write("-" * 60 + "\n\n")
        
        # 統計分析部分
        f.write("="*80 + "\n")
        f.write("錯誤統計分析\n")
        f.write("="*80 + "\n\n")
        
        # 1. 按真實標籤統計錯誤
        f.write("1. 各類別錯誤統計 (按真實標籤)\n")
        f.write("-" * 40 + "\n")
        error_by_true_label = {}
        for idx in misclassified_indices:
            true_label = itos[y_true[idx]]
            if true_label not in error_by_true_label:
                error_by_true_label[true_label] = 0
            error_by_true_label[true_label] += 1
        
        # 計算每個類別在驗證集中的總數
        true_label_counts = {}
        for label_idx in y_true:
            label = itos[label_idx]
            true_label_counts[label] = true_label_counts.get(label, 0) + 1
        
        for true_label in sorted(error_by_true_label.keys()):
            error_count = error_by_true_label[true_label]
            total_count = true_label_counts[true_label]
            error_rate = error_count / total_count * 100
            f.write(f"  {true_label}: {error_count}/{total_count} ({error_rate:.1f}%)\n")
        
        # 2. 混淆矩陣風格的錯誤統計
        f.write(f"\n2. 錯誤分類混淆統計\n")
        f.write("-" * 40 + "\n")
        confusion_errors = {}
        for idx in misclassified_indices:
            true_label = itos[y_true[idx]]
            pred_label = itos[y_pred[idx]]
            
            if true_label not in confusion_errors:
                confusion_errors[true_label] = {}
            if pred_label not in confusion_errors[true_label]:
                confusion_errors[true_label][pred_label] = 0
            confusion_errors[true_label][pred_label] += 1
        
        for true_label in sorted(confusion_errors.keys()):
            f.write(f"  真實標籤: {true_label}\n")
            pred_dict = confusion_errors[true_label]
            for pred_label in sorted(pred_dict.keys(), key=lambda x: pred_dict[x], reverse=True):
                count = pred_dict[pred_label]
                f.write(f"    → 被誤認為 {pred_label}: {count} 次\n")
            f.write("\n")
        
        # 3. 信心度分析
        f.write("3. 錯誤預測信心度分析\n")
        f.write("-" * 40 + "\n")
        error_confidences = [np.max(probabilities[idx]) for idx in misclassified_indices]
        f.write(f"  平均信心度: {np.mean(error_confidences):.4f}\n")
        f.write(f"  最高信心度: {np.max(error_confidences):.4f}\n")
        f.write(f"  最低信心度: {np.min(error_confidences):.4f}\n")
        f.write(f"  標準差: {np.std(error_confidences):.4f}\n")
        
        # 信心度區間統計
        confidence_ranges = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        f.write(f"\n  信心度區間分佈:\n")
        for low, high in confidence_ranges:
            count = sum(1 for conf in error_confidences if low <= conf < high)
            percentage = count / len(error_confidences) * 100
            f.write(f"    {low:.1f}-{high:.1f}: {count} 個 ({percentage:.1f}%)\n")
        
        # 4. 最容易混淆的類別對
        f.write(f"\n4. 最容易混淆的類別對 (Top 10)\n")
        f.write("-" * 40 + "\n")
        confusion_pairs = []
        for true_label, pred_dict in confusion_errors.items():
            for pred_label, count in pred_dict.items():
                confusion_pairs.append((count, true_label, pred_label))
        
        confusion_pairs.sort(reverse=True)
        for i, (count, true_label, pred_label) in enumerate(confusion_pairs[:10]):
            f.write(f"  {i+1:2d}. {true_label} → {pred_label}: {count} 次\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("報告結束\n")
        f.write("="*80 + "\n")
    
    print(f"✅ 詳細錯誤分類報告已保存至: {report_path}")
    return report_path

# 修改原有的 show_misclassified_images 函數，集成txt報告功能
def show_misclassified_images_with_report(dataset, display_dataset, val_indices, y_true, y_pred, itos, 
                                        outputs, max_images=20, save_dir="misclassified_images"):
    """
    顯示並保存辨識錯誤的圖片，同時生成詳細的txt報告
    """
    
    # 首先生成txt報告
    report_path = save_misclassification_report(dataset, display_dataset, val_indices, y_true, y_pred, itos, 
                                              outputs, save_dir)
    
    # 然後執行原有的圖片顯示功能
    misclassified_indices = np.where(y_true != y_pred)[0]
    
    if len(misclassified_indices) == 0:
        print("🎉 所有圖片都被正確分類了！")
        return report_path
    
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
        
        # 獲取預測信心度
        predictions = outputs.predictions
        probabilities = torch.softmax(torch.tensor(predictions), dim=1).numpy()
        confidence = np.max(probabilities[idx])
        
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(image)
        plt.title(f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.3f}", 
                 fontsize=10, color='red')
        plt.axis('off')
        
        # 保存個別圖片
        save_path = os.path.join(save_dir, f"error_{i+1:03d}_true_{true_label}_pred_{pred_label}_conf_{confidence:.3f}.png")
        plt.imsave(save_path, image)
    
    plt.tight_layout()
    overview_path = os.path.join(save_dir, f"misclassified_overview_{safe_model_name}_{timestamp}.png")
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ 錯誤分類圖片概覽已保存至: {overview_path}")
    print(f"✅ 個別圖片已保存至目錄: {save_dir}")
    
    return report_path


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

# 在你的代碼最後部分，將原來的函數調用替換為：
print("\n" + "="*80)
print("開始錯誤分析...")
print("="*80)

# 執行增強版的錯誤分析（包含txt報告）
report_file = show_misclassified_images_with_report(
    dataset, display_dataset, val_indices, y_true, y_pred, itos, outputs,
    max_images=20, save_dir=f"misclassified_{safe_model_name}_{timestamp}"
)

analyze_prediction_confidence(outputs, y_true, y_pred, itos, threshold=0.8)

print(f"\n📄 完整錯誤分析報告已保存至: {report_file}")
print("你可以用文本編輯器打開該文件來查看所有錯誤的詳細信息。")