import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
from torchvision import datasets, transforms
from transformers import AutoModelForImageClassification
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import numpy as np
import random
import csv
import copy
from collections import Counter
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# ====================== 環境變數設定 ======================
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
    return avg_loss, overall_acc, macro_acc, None, None

# ====================== Gaussian KL for VCL ======================
def gaussian_kl_divergence(mu1, logvar1, mu2, logvar2):
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    kl = 0.5 * torch.sum(
        logvar2 - logvar1 + (var1 + (mu1 - mu2).pow(2)) / var2 - 1.0
    )
    return kl

# ====================== 參數設定 ======================
input_size = 384
output_classes = 18
epochs_set1 = 100
epochs_other = epochs_set1
effective_batch_size = 16
gradient_accumulation_steps = 1
learning_rate_other = 0.03
warmup_steps_other = 120
weight_decay = 1e-4
max_grad_norm = 1.0
output_dir = "Swin_VCL_regions"
use_vcl = True
vcl_batch_size = 16
global_class_map = {f'{i:02d}': i for i in range(18)}
seed = 42
train_set1 = True
name = "VCL"

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

    train_loader = DataLoader(train_subset, batch_size=effective_batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=effective_batch_size, shuffle=False)

    model = AutoModelForImageClassification.from_pretrained(
        #"microsoft/resnet-34",
        #"google/vit-large-patch32-384",
        "microsoft/swin-large-patch4-window12-384",
        num_labels=output_classes,
        ignore_mismatched_sizes=True
    )
    nn.init.xavier_uniform_(model.classifier.weight)
    nn.init.zeros_(model.classifier.bias)

    # === 重要優化：啟用 Gradient Checkpointing ===
    model.gradient_checkpointing_enable()

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate_other, momentum=0.9, weight_decay=weight_decay)
    total_steps = epochs_set1 * (len(train_subset) // effective_batch_size)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps_other, t_total=total_steps, base_lr=learning_rate_other)
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
for i in range(1, 8):#range(1, 8):
    datasets_dict[f'set{i}'] = RemappedImageFolder(f'train/set{i}/', transform=transform, global_class_map=global_class_map)

train_subsets = {}
val_subsets = {}
for i in range(1, 8):#range(1, 8):
    dataset = datasets_dict[f'set{i}']
    all_indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(all_indices, test_size=0.1, stratify=dataset.global_labels, random_state=seed)
    train_subsets[f'set{i}'] = Subset(dataset, train_idx)
    val_subsets[f'set{i}'] = Subset(dataset, val_idx)

# ====================== Incremental Training with VCL + Replay ======================
old_model = AutoModelForImageClassification.from_pretrained(
    os.path.join(output_dir, "best"),
    num_labels=output_classes,
    ignore_mismatched_sizes=True
).to('cpu').eval()
old_model.gradient_checkpointing_enable()

old_mu = {}
old_logvar = {}
for name, param in old_model.named_parameters():
    if param.requires_grad:
        old_mu[name] = param.data.clone().to('cpu')
        old_logvar[name] = torch.full_like(param.data, -5.0).to('cpu')

for set_num in range(2,8):#range(2, 8):
    set_name = f'set{set_num}'
    print(f"\n=== Training on {set_name} with VCL + Replay (1000 samples) ===")

    csv_file = os.path.join(output_dir, f"training_metrics_{set_name}.csv")
    header = ['Epoch', 'Train_Loss', 'Train_Accuracy']
    for i in range(1, set_num + 1):
        header.extend([f'Set{i}_Val_Loss', f'Set{i}_Val_Acc'])
    for i in range(1, set_num + 1):
        header.append(f'Macro_Set{i}')
    header.append('Training_Duration (s)')
    with open(csv_file, 'w', newline='') as f:
        csv.writer(f).writerow(header)

    train_loader = DataLoader(train_subsets[set_name], batch_size=effective_batch_size, shuffle=True)
    val_loaders = {f'set{i}': DataLoader(val_subsets[f'set{i}'], batch_size=effective_batch_size, shuffle=False)
                   for i in range(1, set_num + 1)}

    model = copy.deepcopy(old_model).to(device)
    model.gradient_checkpointing_enable()
    model.train()

    model.logvar_dict = {}
    for name, param in list(model.named_parameters()):
        if param.requires_grad:
            logvar_param = nn.Parameter(torch.full_like(param.data, -5.0, device=device))
            model.logvar_dict[name] = logvar_param
            model.register_parameter(f"logvar_{name.replace('.', '_')}", logvar_param)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate_other, momentum=0.9, weight_decay=weight_decay)
    num_steps = epochs_other * (len(train_subsets[set_name]) // effective_batch_size)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps_other, t_total=num_steps, base_lr=learning_rate_other)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()

    # ====================== VCL Replay Coreset - 總數 1000 張 ======================
    replay_iter = None
    replay_loader = None
    if set_num > 1:
        old_datasets = [datasets_dict[f'set{i}'] for i in range(1, set_num)]
        old_dataset = ConcatDataset(old_datasets)
        num_replay = min(1000, len(old_dataset))
        sampled_indices = random.sample(range(len(old_dataset)), num_replay)
        replay_dataset = Subset(old_dataset, sampled_indices)
        replay_loader = DataLoader(
            replay_dataset,
            batch_size=vcl_batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False
        )
        replay_iter = iter(replay_loader)
        print(f"✅ Replay coreset 已建立：共 {num_replay} 張圖片（來自前 {set_num-1} 個 set）")
    # ================================================================================

    start_time = time.time()
    best_val_acc = 0.0

    for epoch in range(epochs_other):
        model.train()
        train_loss = train_correct = train_total = 0.0
        train_bar = tqdm(train_loader, desc=f'{set_name} Epoch {epoch+1}/{epochs_other} [VCL+Replay]', leave=False)

        for step, (inputs, labels) in enumerate(train_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                # ====================== MC sampling + local reparam trick ======================
                orig_weight = model.classifier.weight.data.clone()
                orig_bias = model.classifier.bias.data.clone()

                logvar_w = model.logvar_dict['classifier.weight']
                sigma_w = torch.exp(0.5 * logvar_w)
                eps_w = torch.randn_like(model.classifier.weight)
                sampled_weight = model.classifier.weight + sigma_w * eps_w

                logvar_b = model.logvar_dict['classifier.bias']
                sigma_b = torch.exp(0.5 * logvar_b)
                eps_b = torch.randn_like(model.classifier.bias)
                sampled_bias = model.classifier.bias + sigma_b * eps_b

                model.classifier.weight.data = sampled_weight
                model.classifier.bias.data = sampled_bias

                # Current task forward
                outputs = model(inputs).logits
                ce_loss = criterion(outputs, labels)

                # ====================== Replay forward ======================
                replay_ce = 0.0
                if replay_iter is not None:
                    try:
                        r_inputs, r_labels = next(replay_iter)
                    except StopIteration:
                        replay_iter = iter(replay_loader)
                        r_inputs, r_labels = next(replay_iter)
                    r_inputs = r_inputs.to(device)
                    r_labels = r_labels.to(device)
                    r_outputs = model(r_inputs).logits
                    replay_ce = criterion(r_outputs, r_labels)
                    ce_loss = (ce_loss + replay_ce) / 2.0   # 平衡 current + replay

                # 恢復原始 weights
                model.classifier.weight.data = orig_weight
                model.classifier.bias.data = orig_bias

                # ====================== KL divergence ======================
                loss2 = 0.0
                if use_vcl:
                    kl_loss = 0.0
                    for name, param in model.named_parameters():
                        if param.requires_grad and name in old_mu:
                            mu1 = param
                            logvar1 = model.logvar_dict[name]
                            mu2 = old_mu[name].to(device)
                            logvar2 = old_logvar[name].to(device)
                            kl = gaussian_kl_divergence(mu1.view(-1), logvar1.view(-1), mu2.view(-1), logvar2.view(-1))
                            kl_loss += kl
                    N_task = len(train_subsets[set_name])
                    loss2 = kl_loss / N_task

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

        # ====================== Validation ======================
        val_results = {}
        macro_per_set = []
        for i in range(1, set_num + 1):
            s = f'set{i}'
            v_loss, v_acc, m_acc, _, _ = evaluate_model(model, val_loaders[s], criterion, device, f'Val {s}')
            val_results[s] = {'loss': v_loss, 'accuracy': v_acc}
            macro_per_set.append(m_acc)

        overall_macro = sum(macro_per_set) / len(macro_per_set) if macro_per_set else 0.0

        if overall_macro > best_val_acc:
            best_val_acc = overall_macro
            model.save_pretrained(os.path.join(output_dir, f'best_{set_name}'))
            print(f'##### Saved best for {set_name} with Overall Macro: {best_val_acc:.4f} #####')

        duration = time.time() - start_time

        print(f'{set_name} Epoch [{epoch+1}/{epochs_other}] Train Acc: {train_acc:.4f}')
        for i in range(1, set_num + 1):
            s = f'set{i}'
            print(f' Set{i}_Val_Acc: {val_results[s]["accuracy"]:.4f} | Macro_Set{i}: {macro_per_set[i-1]:.4f}')

        # CSV 寫入
        row = [epoch + 1, round(train_loss,6), round(train_acc,6)]
        for i in range(1, set_num + 1):
            s = f'set{i}'
            row.extend([round(val_results[s]['loss'],6), round(val_results[s]['accuracy'],6)])
        for m in macro_per_set:
            row.append(round(m,6))
        row.append(round(duration,2))
        with open(csv_file, 'a', newline='') as f:
            csv.writer(f).writerow(row)

    model.save_pretrained(os.path.join(output_dir, f'last_{set_name}'))

    # 更新 prior (q_{t-1})
    best_path = os.path.join(output_dir, f'best_{set_name}')
    if os.path.exists(best_path):
        old_model = AutoModelForImageClassification.from_pretrained(
            best_path, num_labels=output_classes, ignore_mismatched_sizes=True
        ).to('cpu').eval()
        old_model.gradient_checkpointing_enable()
        for name, param in old_model.named_parameters():
            if param.requires_grad:
                old_mu[name] = param.data.clone().to('cpu')
                if name in model.logvar_dict:
                    old_logvar[name] = model.logvar_dict[name].data.clone().to('cpu')
                else:
                    old_logvar[name] = torch.full_like(param.data, -5.0).to('cpu')

    torch.cuda.empty_cache()

print("✅ Incremental training with VCL + 1000 Replay Coreset 完成！")