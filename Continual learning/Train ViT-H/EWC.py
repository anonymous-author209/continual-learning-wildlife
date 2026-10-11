import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
from torchvision import datasets, transforms
from transformers import ViTHybridForImageClassification   
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import numpy as np
import random
import csv
import copy
from collections import Counter
from torch.amp import autocast #, GradScaler
from torch.utils.tensorboard import SummaryWriter

# ====================== Environment Variables ======================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
torch.backends.cudnn.benchmark = False

# ====================== WarmupCosineSchedule ======================
class WarmupCosineSchedule:
    def __init__(self, optimizer, warmup_steps, t_total, base_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.base_lr = base_lr
        self.current_step = 0
    def step(self):
        self.current_step += 1
        lr = self.get_lr()[0]
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    def get_lr(self):
        if self.current_step < self.warmup_steps:
            return [self.current_step / self.warmup_steps * self.base_lr]
        progress = (self.current_step - self.warmup_steps) / (self.t_total - self.warmup_steps)
        return [0.5 * self.base_lr * (1 + np.cos(np.pi * progress))]

# ====================== RemappedImageFolder ======================
class RemappedImageFolder(Dataset):
    def __init__(self, root, transform=None, global_class_map=None):
        self.dataset = datasets.ImageFolder(root, transform=transform)
        self.global_class_map = global_class_map
        self.local_to_global = {}
     
        for cls in self.dataset.classes:
            try:
                local_label = int(cls)
                self.local_to_global[local_label] = global_class_map[cls]
            except ValueError:
                raise ValueError(f"Class name {cls} cannot be converted to integer")
     
        new_targets = [int(self.dataset.classes[target]) for target in self.dataset.targets]
        self.dataset.targets = new_targets
        self.global_labels = [self.local_to_global[local_label] for local_label in self.dataset.targets]
     
        self.label_dist = Counter(self.global_labels)
        print(f"Loaded {len(self.dataset)} samples from {root}")
        print(f"Global label distribution: {self.label_dist}")
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return img, self.global_labels[idx]

# ====================== Evaluate Model ======================
def evaluate_model(model, data_loader, criterion, device, desc="Evaluation"):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    class_correct = {i: 0 for i in range(18)}
    class_total = {i: 0 for i in range(18)}
 
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc=desc, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast(device_type='cuda'):
                outputs = model(inputs).logits
                loss = criterion(outputs, labels)
         
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
         
            for label, pred in zip(labels, predicted):
                lbl = label.item()
                class_total[lbl] += 1
                if label == pred:
                    class_correct[lbl] += 1
 
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
 
    per_class_acc = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0 for i in range(18)]
    macro_acc = sum(per_class_acc) / len(per_class_acc) if per_class_acc else 0.0
 
    class_acc_dict = {i: class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0 for i in range(18)}
 
    return avg_loss, overall_acc, macro_acc, class_acc_dict, class_total

# ====================== EWC function ======================
def compute_fisher_information(model, data_loader, device, num_samples=512):
    """ Fisher Information Matrix """
    model.eval()
    for param in model.parameters():
        param.requires_grad = True
  
    fisher = {n: torch.zeros_like(p, device='cpu') for n, p in model.named_parameters() if p.requires_grad}
  
    criterion = nn.CrossEntropyLoss()
    samples_processed = 0
  
    progress_bar = tqdm(total=num_samples, desc="Computing Fisher Information", leave=False)
  
    for inputs, labels in data_loader:
        if samples_processed >= num_samples:
            break
          
        inputs, labels = inputs.to(device), labels.to(device)
      
        model.zero_grad()
      
        with autocast(device_type='cuda'):
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
      
        loss.backward()
      
        for name, param in model.named_parameters():
            if param.grad is not None and name in fisher:
                fisher[name] += (param.grad.data ** 2).cpu()
      
        samples_processed += inputs.size(0)
        progress_bar.update(inputs.size(0))
  
    progress_bar.close()
  
    for name in fisher:
        fisher[name] /= min(samples_processed, num_samples)
  
    print(f"Fisher Information computed on {samples_processed} samples.")
    return fisher

def ewc_loss(model, old_model, fisher_dict, ewc_lambda=5000.0):
    """EWC Regularization Loss"""
    loss_reg = 0.0
    for (name, param), (_, old_param) in zip(model.named_parameters(), old_model.named_parameters()):
        if name in fisher_dict and param.requires_grad:
            fisher = fisher_dict[name].to(param.device)
            diff = param - old_param.to(param.device)
            loss_reg += (fisher * diff ** 2).sum()
  
    return ewc_lambda * loss_reg

# ====================== Parameter Settings ======================
input_size = 384
output_classes = 18
epochs_set1 = 100
epochs_other = epochs_set1
batch_size = 16
effective_batch_size = 16
gradient_accumulation_steps = 1
learning_rate_set1 = 0.03
learning_rate_other = 0.03
warmup_steps_set1 = 120
warmup_steps_other = 120
weight_decay = 1e-4
max_grad_norm = 1.0
pretrained_model_name = "google/vit-hybrid-base-bit-384"
output_dir = "VITHybird_EWC_regions"
name = "EWC"
use_ewc = True
ewc_lambda = 5000.0
fip_batch_size = 16
global_class_map = {f'{i:02d}': i for i in range(18)}
seed = 42
train_set1 = False
# ====================== Random Seed ======================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

# ====================== Data Transforms ======================
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ====================== Training on set1 ======================
if train_set1:
    print("\n=== Training on set1 ===")
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs", name + "_set1"))
    csv_file_set1 = os.path.join(output_dir, "training_metrics_set1.csv")
    with open(csv_file_set1, 'w', newline='') as f:
        csv.writer(f).writerow(['Epoch', 'Train_Loss', 'Train_Accuracy',
                                'Val_Loss', 'Val_Accuracy', 'Macro_Set1',
                                'Training_Duration (s)'])
    train_dataset = RemappedImageFolder('train/set1/', transform=transform, global_class_map=global_class_map)
    all_indices = list(range(len(train_dataset)))
    labels = train_dataset.global_labels
    train_idx, val_idx = train_test_split(all_indices, test_size=0.1, stratify=labels, random_state=seed)
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    # === ViTHybridForImageClassification ===
    model = ViTHybridForImageClassification.from_pretrained(
        pretrained_model_name,
        num_labels=output_classes,
        ignore_mismatched_sizes=True
    )
    nn.init.xavier_uniform_(model.classifier.weight)
    nn.init.zeros_(model.classifier.bias)
  
    model.gradient_checkpointing_enable()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate_set1, momentum=0.9, weight_decay=weight_decay)
    total_steps = epochs_set1 * (len(train_subset) // batch_size)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps_set1, t_total=total_steps, base_lr=learning_rate_set1)
    scaler = torch.amp.GradScaler('cuda')
    start_time = time.time()
    best_val_acc = 0.0
    for epoch in range(epochs_set1):
        model.train()
        train_loss = train_correct = train_total = 0.0
        train_bar = tqdm(train_loader, desc=f'set1 Epoch {epoch+1}/{epochs_set1} [Train]', leave=False)
     
        for step, (inputs, labels) in enumerate(train_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
         
            with autocast(device_type='cuda'):
                outputs = model(inputs).logits
                loss = criterion(outputs, labels)
         
            scaler.scale(loss).backward()
            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
         
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
     
        train_loss /= train_total
        train_acc = train_correct / train_total
        val_loss, val_acc, macro_acc, _, _ = evaluate_model(model, val_loader, criterion, device, 'set1 [Val]')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(os.path.join(output_dir, 'best'))
            print(f'##### Saved best model for set1 at epoch {epoch+1} with Val Acc: {best_val_acc:.4f} #####')
        duration = time.time() - start_time
        with open(csv_file_set1, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, round(train_loss,6), round(train_acc,6),
                                    round(val_loss,6), round(val_acc,6), round(macro_acc,6), round(duration,2)])
        print(f'set1 Epoch [{epoch+1}/{epochs_set1}] Val Acc: {val_acc:.4f} | Macro_Set1: {macro_acc:.4f}')
    model.save_pretrained(os.path.join(output_dir, 'last_set1'))
    writer.close()
    torch.cuda.empty_cache()

# ====================== Load all datasets ======================
datasets_dict = {}
for i in range(1, 8):
    datasets_dict[f'set{i}'] = RemappedImageFolder(f'train/set{i}/', transform=transform, global_class_map=global_class_map)

train_subsets = {}
val_subsets = {}
for i in range(1, 8):
    dataset = datasets_dict[f'set{i}']
    all_indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(all_indices, test_size=0.1, stratify=dataset.global_labels, random_state=seed)
    train_subsets[f'set{i}'] = Subset(dataset, train_idx)
    val_subsets[f'set{i}'] = Subset(dataset, val_idx)

# ====================== Incremental Training with EWC ======================
# === ViTHybridForImageClassification ===
old_model = ViTHybridForImageClassification.from_pretrained(
    os.path.join(output_dir, "best"),
    num_labels=output_classes,
    ignore_mismatched_sizes=True
).to('cpu').eval()

old_model.gradient_checkpointing_enable()

# === Fisher Information ===
if use_ewc:
    print("\n=== Computing Fisher Information for EWC (based on set1) ===")
    fisher_loader = DataLoader(datasets_dict['set1'], batch_size=fip_batch_size, shuffle=True, num_workers=4, pin_memory=True)
  
    old_model = old_model.to(device)
    fisher_dict = compute_fisher_information(old_model, fisher_loader, device, num_samples=512)
    old_model = old_model.to('cpu')
    torch.cuda.empty_cache()

for set_num in range(2, 8):
    set_name = f'set{set_num}'
    print(f"\n=== Training on {set_name} with EWC ===")
 
    csv_file = os.path.join(output_dir, f"training_metrics_{set_name}.csv")
 
    header = ['Epoch', 'Train_Loss', 'Train_Accuracy']
    for i in range(1, set_num + 1):
        header.extend([f'Set{i}_Val_Loss', f'Set{i}_Val_Acc'])
    for i in range(1, set_num + 1):
        header.append(f'Macro_Set{i}')
    header.append('Training_Duration (s)')
 
    with open(csv_file, 'w', newline='') as f:
        csv.writer(f).writerow(header)
    
    # Replay (1000 samples)
    all_prev_train = ConcatDataset([train_subsets[f'set{i}'] for i in range(1, set_num)])
    replay_size = 1000
    num_prev = len(all_prev_train)
    if num_prev > replay_size:
        replay_indices = random.sample(range(num_prev), replay_size)
    else:
        replay_indices = list(range(num_prev))
    replay_dataset = Subset(all_prev_train, replay_indices)
    combined_train_dataset = ConcatDataset([train_subsets[set_name], replay_dataset])
    train_loader = DataLoader(combined_train_dataset, batch_size=effective_batch_size, shuffle=True)
    
    val_loaders = {f'set{i}': DataLoader(val_subsets[f'set{i}'], batch_size=effective_batch_size, shuffle=False)
                   for i in range(1, set_num + 1)}
    
    model = copy.deepcopy(old_model).to(device)
    model.gradient_checkpointing_enable()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate_other, momentum=0.9, weight_decay=weight_decay)
    num_steps = epochs_other * (len(train_subsets[set_name]) // effective_batch_size)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps_other, t_total=num_steps, base_lr=learning_rate_other)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()
    best_val_acc = 0.0
    for epoch in range(epochs_other):
        model.train()
        train_loss = train_correct = train_total = 0.0
        train_bar = tqdm(train_loader, desc=f'{set_name} Epoch {epoch+1}/{epochs_other} [Train]', leave=False)
     
        for step, (inputs, labels) in enumerate(train_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
         
            with autocast(device_type='cuda'):
                outputs = model(inputs).logits
                ce_loss = criterion(outputs, labels)
                loss2 = ewc_loss(model, old_model, fisher_dict, ewc_lambda) if use_ewc else 0.0
              
                loss = ce_loss + loss2
            scaler.scale(loss).backward()
            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            train_loss += loss.item() * inputs.size(0)
            train_total += labels.size(0)
            train_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            if step % 4 == 0:
                torch.cuda.empty_cache()
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation
        val_results = {}
        macro_per_set = []
        for i in range(1, set_num + 1):
            s = f'set{i}'
            v_loss, v_acc, m_acc, _, _ = evaluate_model(model, val_loaders[s], criterion, device, f'Val {s}')
            val_results[s] = {'loss': v_loss, 'accuracy': v_acc}
            macro_per_set.append(m_acc)
        overall_macro = sum(macro_per_set) / len(macro_per_set) if macro_per_set else 0.0
        save_acc = overall_macro if use_ewc else val_results[set_name]['accuracy']
        if save_acc > best_val_acc:
            best_val_acc = save_acc
            model.save_pretrained(os.path.join(output_dir, f'best_{set_name}'))
            print(f'##### Saved best for {set_name} with save_acc: {best_val_acc:.4f} #####')
        duration = time.time() - start_time
        row = [epoch + 1, round(train_loss,6), round(train_acc,6)]
        for i in range(1, set_num + 1):
            s = f'set{i}'
            row.extend([round(val_results[s]['loss'],6), round(val_results[s]['accuracy'],6)])
        for m in macro_per_set:
            row.append(round(m,6))
        row.append(round(duration,2))
        with open(csv_file, 'a', newline='') as f:
            csv.writer(f).writerow(row)
        print(f'{set_name} Epoch [{epoch+1}/{epochs_other}] Train Acc: {train_acc:.4f}')
        for i in range(1, set_num + 1):
            s = f'set{i}'
            print(f' Set{i}_Val_Acc: {val_results[s]["accuracy"]:.4f} | Macro_Set{i}: {macro_per_set[i-1]:.4f}')
    model.save_pretrained(os.path.join(output_dir, f'last_{set_name}'))
    
    # Update the old_model and recompute the Fisher matrix.
    best_path = os.path.join(output_dir, f'best_{set_name}')
    if os.path.exists(best_path) and use_ewc:
        print(f"Updating old_model and recomputing Fisher for next stage...")
        # === ViTHybridForImageClassification ===
        old_model = ViTHybridForImageClassification.from_pretrained(
            best_path, num_labels=output_classes, ignore_mismatched_sizes=True
        ).to('cpu').eval()
        old_model.gradient_checkpointing_enable()
        all_old_datasets = [datasets_dict[f'set{i}'] for i in range(1, set_num + 1)]
        all_old_dataset = ConcatDataset(all_old_datasets)
        sampled_idx = random.sample(range(len(all_old_dataset)), min(512, len(all_old_dataset)))
        fisher_dataset = Subset(all_old_dataset, sampled_idx)
        fisher_loader = DataLoader(fisher_dataset, batch_size=fip_batch_size, shuffle=True, num_workers=4, pin_memory=True)
      
        old_model = old_model.to(device)
        fisher_dict = compute_fisher_information(old_model, fisher_loader, device, num_samples=512)
        old_model = old_model.to('cpu')
        torch.cuda.empty_cache()

print("Incremental training with official EWC + Replay (1000 samples) completed.")