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
matplotlib.use('Agg')  # Must be set before pyplot
import os

import seaborn as sns
import datetime
from PIL import Image, ImageFile
import wandb    #CMD wandb login  # Enter your API key

from torchvision.datasets import ImageFolder
from torchvision.transforms import Normalize, Resize, ToTensor, Compose
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter
import torchvision.transforms as transforms

from transformers import TrainingArguments, Trainer
from transformers import AutoImageProcessor, ResNetForImageClassification

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# Ignore image corruption errors
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define a safe image reading Dataset class
class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except (OSError, RuntimeError, ValueError) as e:
            print(f"❗ Skipping broken image at index {index}: {e}")
            return self.__getitem__((index + 1) % len(self))
        
# Generate timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

# ✅ Initialize wandb (will restart for each hyperparameter search)
wandb.init(
  project="fine-tuning",     # ✅ Can share the same project
  name="resnet-34-384-mis",  # ✅ Change this to distinguish different models
  group="resnet",      # ✅ Can add group for easier comparison
  reinit=True
)
# ==============================================================================
# -- Check GPU Availability ----------------------------------------------------
# ==============================================================================

# # Check if GPU is available, raise error if not
# if not torch.cuda.is_available():
#     raise RuntimeError("GPU is not available, please check if CUDA is installed correctly!")

# device = torch.device("cuda")  # Force use of GPU


# ==============================================================================
# -- Dataset Setup -------------------------------------------------------------
# ==============================================================================

dataset_path = "/path/to/dataset"   #origin dataset_path = "/mnt/g/animal-12K" /media/username/F2/

# G:\roboflow_ori\4.8K_vit
# G:\roboflow_ori\7.2K

# ==============================================================================
# -- Image Preprocessing -------------------------------------------------------
# ==============================================================================

# Set up image processing

# google/vit-large-patch16-224  Imagenet-1K 
# google/vit-base-patch16-224-in21k Imagenet-21K

# Currently used:
#   B16-224、L16-224、L16-384
#   18、34、50、152

model_name = "microsoft/resnet-34"
local_config_path = "/path/to/resnet-34" 
#   #1) For local run (if need)

# resnet 18 snapshots/65a5785d9156231087c481e0c7dd33a5ff6f7e3e
# restnet 34    4d180fbdc81b4d5d4eaa86ab89bdd669200dd477
# restnet 50    34c2154c194f829b11125337b98c8f5f9965ff19
# restnet 152   189229b1590873cc46c03813abc7d0a7cb8f02b2
processor = AutoImageProcessor.from_pretrained(local_config_path) #model_name   Originally: ViTImageProcessor、AutoFeatureExtractor
mu, sigma = processor.image_mean, processor.image_std
size = processor.size
# processor.size = {"height": 384, "width": 384}

# Set up transforms
transform = Compose([
    Resize((384, 384)),  # ResNet input size    
    # RandomCrop(224, padding=20),  # ✅ Only do slight random crop to avoid losing the target
    # CenterCrop(224),  # ✅ Keep the target intact
    # RandomHorizontalFlip(p=0.5),  # ✅ Horizontal flip
    # RandomRotation(degrees=5),  # ✅ Limit rotation range to prevent excessive deformation
    # ColorJitter(brightness=0.05, contrast=0.05),  # ✅ Avoid excessive lighting changes
    ToTensor(),
    Normalize(mean=mu, std=sigma)
])

# Set up display transform (without normalization)
display_transform = Compose([
    Resize((384, 384)),
    ToTensor()
])

# Load custom dataset
dataset = ImageFolder(root=dataset_path, transform=transform)
# Create dataset for display (without normalization)
display_dataset = ImageFolder(root=dataset_path, transform=display_transform)

# Create label mappings
itos = {i: label for i, label in enumerate(dataset.classes)}
stoi = {label: i for i, label in itos.items()}

# Split train & test sets
train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
trainds = torch.utils.data.Subset(dataset, train_indices)
valds = torch.utils.data.Subset(dataset, val_indices)
# Validation set for display
display_valds = torch.utils.data.Subset(display_dataset, val_indices)

def collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])  # Images
    labels = torch.tensor([item[1] for item in batch])  # Labels
    return {"pixel_values": pixel_values, "labels": labels}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# -- Model Setup ---------------------------------------------------------------
# ==============================================================================

model = ResNetForImageClassification.from_pretrained(  # ViTHybridForImageClassification, originally: ViTForImageClassification
    local_config_path,
    num_labels=len(itos),
    id2label=itos,
    label2id=stoi,
    ignore_mismatched_sizes=True
).to(device)  # Ensure model is moved to GPU

# # ✅ Freeze the first few layers of ResNet-50 (only train the last few layers)
# for param in model.base_model.parameters():
#     param.requires_grad = False  # ✅ Freeze most layers, do not update weights

# # ✅ Only unfreeze the classification layer (last few layers)
# for param in model.classifier.parameters():
#     param.requires_grad = True

# ==============================================================================
# -- Training Configuration ----------------------------------------------------
# ==============================================================================

args = TrainingArguments(
    output_dir="./results-{model_name}",  # ✅ Change to external drive
    save_strategy="epoch",
    evaluation_strategy="epoch",  #evaluation_strategy HF(v 4.40.2, New version :save_strategy)
    logging_strategy="epoch",  # Ensure train_loss is logged
    learning_rate=2e-5, # resnet-lower learning rate (5e-6), use cosine scheduler  Originally: learning_rate=2e-5
    # lr_scheduler_type="cosine",  # resnet-use Cosine scheduler
    # warmup_ratio=0.1,  #  Let learning rate rise slowly in the first 10% of steps
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir='logs',
    fp16=torch.cuda.is_available(),  # Enable mixed precision training (if GPU available)
    
    report_to="wandb",   # ✅ Log with wandb
    run_name=f"fine-tune-{model_name}",
)


# Set up Trainer
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
trainer.save_model(f"fine-tuned-animal-18K-{model_name}")  # Save model
trainer.evaluate()


# ==============================================================================
# -- Logging & Visualization ---------------------------------------------------
# ==============================================================================

# Extract loss and accuracy records
# In Hugging Face Trainer, the TrainOutput object does not contain the training_loss_history attribute.
train_steps = [log["epoch"] for log in trainer.state.log_history if "loss" in log]
eval_steps = [log["epoch"] for log in trainer.state.log_history if "eval_loss" in log]     

loss_values = [log["loss"] for log in trainer.state.log_history if "loss" in log]
eval_loss_values = [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log]
eval_accuracy_values = [log["eval_accuracy"] for log in trainer.state.log_history if "eval_accuracy" in log]

# Append timestamp and model name to output filenames
safe_model_name = model_name.replace("/", "-")  # Avoid FileNotFoundError: [Errno 2] No such file or directory
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

# Evaluate model
outputs = trainer.predict(valds)
y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)

# ==============================================================================
# -- confusion_matrix ---------------------------------------------------------
# ==============================================================================

# Plot confusion matrix  (cm)
# cm = confusion_matrix(y_true, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(itos.values()))
# disp.plot(xticks_rotation=45)
# plt.show()

# Plot confusion matrix  (Normalized cm)
def plot_normalized_confusion_matrix(y_true, y_pred, save_path=f'normalized_confusion_matrix_{safe_model_name}_{timestamp}.png'):
    cm = confusion_matrix(y_true, y_pred, normalize='true')  #  Normalize to 0~1
    
    plt.figure(figsize=(16, 8))
    sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=list(itos.values()),
            yticklabels=list(itos.values())
    )
    # Add labels and title
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f'Normalized Confusion Matrix_{safe_model_name}')
    # Automatically adjust layout
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save image
    plt.savefig(save_path, dpi=300)
    plt.close()  # Prevent memory accumulation or unreleased figures
    print(f"Normalized Confusion Matrix saved to {save_path}")

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Model accuracy on validation set: {accuracy:.4f}")
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

# Get F1-score and plot
f1_scores = {label: metrics["f1-score"] for label, metrics in report.items() if isinstance(metrics, dict)}
plot_f1_score_curve(f1_scores)


# ==============================================================================
# -- Error Analysis - Display Misclassified Images ----------------------------
# ==============================================================================

def show_misclassified_images(display_dataset, val_indices, y_true, y_pred, itos, 
                            max_images=20, save_dir="misclassified_images"):
    """
    Display and save misclassified images
    
    Args:
        display_dataset: Dataset for display (without normalization)
        val_indices: Validation set indices
        y_true: True labels
        y_pred: Predicted labels
        itos: Index-to-label mapping
        max_images: Maximum number of images to display
        save_dir: Directory to save misclassified images
    """
    
    # Find indices of misclassified samples
    misclassified_indices = np.where(y_true != y_pred)[0]
    
    if len(misclassified_indices) == 0:
        print("🎉 All images were correctly classified!")
        return
    
    print(f"❌ Total of {len(misclassified_indices)} misclassified images found")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Limit the number of images to display
    show_indices = misclassified_indices[:max_images]
    
    # Calculate subplot layout
    n_images = len(show_indices)
    n_cols = 4  # 4 images per row
    n_rows = (n_images + n_cols - 1) // n_cols
    
    plt.figure(figsize=(20, 5 * n_rows))
    
    for i, idx in enumerate(show_indices):
        # Get the original index in the validation set
        original_idx = val_indices[idx]
        image, _ = display_dataset[original_idx]
        
        # Convert tensor to displayable format
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
            image = np.clip(image, 0, 1)
        
        true_label = itos[y_true[idx]]
        pred_label = itos[y_pred[idx]]
        
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(image)
        plt.title(f"True: {true_label}\nPred: {pred_label}", fontsize=10, color='red')
        plt.axis('off')
        
        # Save individual images
        save_path = os.path.join(save_dir, f"error_{i+1:03d}_true_{true_label}_pred_{pred_label}.png")
        plt.imsave(save_path, image)
    
    plt.tight_layout()
    overview_path = os.path.join(save_dir, f"misclassified_overview_{safe_model_name}_{timestamp}.png")
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Misclassified images overview saved to: {overview_path}")
    print(f"✅ Individual images saved to directory: {save_dir}")
    
    # Statistics of error counts per class
    error_stats = {}
    for idx in misclassified_indices:
        true_label = itos[y_true[idx]]
        pred_label = itos[y_pred[idx]]
        
        if true_label not in error_stats:
            error_stats[true_label] = {}
        if pred_label not in error_stats[true_label]:
            error_stats[true_label][pred_label] = 0
        error_stats[true_label][pred_label] += 1
    
    print("\n📊 Misclassification Statistics:")
    for true_label, pred_dict in error_stats.items():
        print(f"\nTrue label: {true_label}")
        for pred_label, count in pred_dict.items():
            print(f"  → Misclassified as {pred_label}: {count} times")

def analyze_prediction_confidence(outputs, y_true, y_pred, itos, threshold=0.5):
    """
    Analyze prediction confidence
    """
    predictions = outputs.predictions
    probabilities = torch.softmax(torch.tensor(predictions), dim=1).numpy()
    
    # Get the highest probability of the prediction
    max_probs = np.max(probabilities, axis=1)
    
    # Find low-confidence predictions
    low_confidence_mask = max_probs < threshold
    misclassified_mask = y_true != y_pred
    
    print(f"\n🎯 Prediction Confidence Analysis:")
    print(f"Total samples: {len(y_true)}")
    print(f"Low confidence predictions (<{threshold}): {np.sum(low_confidence_mask)}")
    print(f"Mispredictions: {np.sum(misclassified_mask)}")
    print(f"Low confidence and incorrect predictions: {np.sum(low_confidence_mask & misclassified_mask)}")
    
    # Plot confidence distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(max_probs[~misclassified_mask], bins=50, alpha=0.7, label='Correct prediction', color='green')
    plt.hist(max_probs[misclassified_mask], bins=50, alpha=0.7, label='Wrong prediction', color='red')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Count')
    plt.title('Prediction Confidence Distribution')
    plt.legend()
    plt.axvline(x=threshold, color='black', linestyle='--', label=f'Threshold {threshold}')
    
    # Correct prediction
    # Wrong prediction
    # Prediction Confidence
    # Count
    # Prediction Confidence Distribution
    # Threshold

    # Prediction Confidence
    # Misclassification
    # Ground Truth Label
    # Predicted Label
    # Confidence Distribution
    # Correct Predictions
    # Incorrect Predictions
    # Low Confidence
    # Threshold
    # Average Confidence per Class


    plt.subplot(1, 2, 2)
    # Calculate average confidence for each class
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
    
    print(f"✅ Prediction confidence analysis plot saved to: {confidence_path}")

# Execute error analysis
print("\n" + "="*80)
print("Starting error analysis...")
print("="*80)

show_misclassified_images(display_dataset, val_indices, y_true, y_pred, itos, 
                         max_images=20, save_dir=f"misclassified_{safe_model_name}_{timestamp}")

analyze_prediction_confidence(outputs, y_true, y_pred, itos, threshold=0.8)