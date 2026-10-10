import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==================== 設定區 ====================
model = './'
method = ''
folders = [
    ""#"/media/lab/EXTERNAL_USB/碩士畢業論文/Swin_class/Swin_FIP_class"
]
method_names = [f"{method}"]
group_names = [f"Group{i}" for i in range(1, 8)]

FIGSIZE = (14, 9)
FONT_SIZE_LABEL = 16
FONT_SIZE_LEGEND = 12
FONT_SIZE_TITLE = 18
LINE_WIDTH = 2.5

# 資料結構
acc_segments = {method: {group: [] for group in group_names} for method in method_names}
task_start_epochs = {method: {} for method in method_names}
global_max_epoch = 0

# 定義圖例顯示文字
legend_labels = {
    "Group1": "Group1",
    "Group2": "Group2",
    "Group3": "Group3",
    "Group4": "Group4",
    "Group5": "Group5",
    "Group6": "Group6",
    "Group7": "Group7",
}
"""
legend_labels = {
    "Group1": "Carnivora",
    "Group2": "Artiodactyla",
    "Group3": "Rodentia",
    "Group4": "Galliformes",
    "Group5": "Columbiformes",
    "Group6": "Primates",
}
"""

# ==================== 讀取資料 ====================
for folder, method in zip(folders, method_names):
    print(f"\n處理方法: {method} ({folder})")
    
    current_global_epoch = 0
    
    for set_idx in range(1, 8):
        file_name = f"training_metrics_set{set_idx}.csv"
        file_path = os.path.join(folder, file_name)
        
        if not os.path.exists(file_path):
            print(f" ✗ 找不到 {file_name}")
            continue
        
        try:
            df = pd.read_csv(file_path)
            n_epochs = len(df)
            
            print(f" Set{set_idx:2d} : {n_epochs:4d} epochs (global {current_global_epoch + 1} ~ {current_global_epoch + n_epochs})")
            
            new_group = f"Group{set_idx}"
            if new_group not in task_start_epochs[method]:
                task_start_epochs[method][new_group] = current_global_epoch + 1
            
            for prev_idx in range(1, set_idx + 1):
                group = f"Group{prev_idx}"
                
                if prev_idx == 1:
                    col = "Val_Accuracy" if "Val_Accuracy" in df.columns else "Set1_Val_Acc" # "Set1_Val_Acc"
                else:
                    col = f"Set{prev_idx}_Val_Acc"
                
                if col in df.columns:
                    values = df[col].to_numpy(dtype=float)
                    if len(values) == n_epochs:
                        acc_segments[method][group].append((current_global_epoch, values))
                        print(f" └─ {group:<8} 收集自 {col:12} ({len(values)} 筆)")
                    else:
                        print(f" ⚠ 長度不匹配 {col} ({len(values)} ≠ {n_epochs})")
                else:
                    print(f" ✗ 找不到欄位：{col}")
            
            current_global_epoch += n_epochs
            global_max_epoch = max(global_max_epoch, current_global_epoch)
        
        except Exception as e:
            print(f" 讀取失敗 {file_name}：{e}")
    
    print(f" → 總 global epochs：{global_max_epoch} (顯示為 1 ~ {global_max_epoch})\n")

# ==================== 繪圖 ====================
fig, ax = plt.subplots(figsize=FIGSIZE)
method = method_names[0]
colors = plt.cm.tab10(np.linspace(0, 0.95, len(group_names)))

for g_idx, group in enumerate(group_names):
    segments = acc_segments[method][group]
    if not segments:
        print(f" {group} 無資料")
        continue
    
    start_epoch_display = task_start_epochs[method].get(group, None)
    if start_epoch_display is None:
        continue
    
    # === 合併成完整曲線 ===
    segments_sorted = sorted(segments, key=lambda s: s[0])
    
    x_full = []
    y_full = []
    
    for seg_start_internal, y_values in segments_sorted:
        x_seg = np.arange(seg_start_internal + 1, seg_start_internal + 1 + len(y_values))
        y_seg = y_values * 100
        x_full.extend(x_seg)
        y_full.extend(y_seg)
    
    if not x_full:
        continue
    
    # 轉成 numpy array 並處理 NaN
    x_full = np.array(x_full)
    y_full = np.array(y_full)
    
    nan_count = np.sum(np.isnan(y_full))
    if nan_count > 0:
        print(f"警告：{group} 在拼接後有 {nan_count} 個 NaN，已使用前後填充")
    
    # 前向 + 後向填充（讓斷開的地方連起來）
    y_series = pd.Series(y_full)
    y_series = y_series.ffill().bfill()  # 先往前填，再往後填，避免頭尾空值
    y_full = y_series.values
    
    # 檢查 x 是否連續（診斷用）
    if len(x_full) > 1:
        is_continuous = np.all(np.diff(x_full) == 1)
        if not is_continuous:
            print(f"注意：{group} 的 x 不完全連續（有跳號），可能導致視覺斷開")
            # 你可以選擇在這裡插值 x，但通常不建議，除非真的有缺
    
    # 一次畫完整曲線
    ax.plot(x_full, y_full,
            label=legend_labels.get(group, group),
            color=colors[g_idx],
            linewidth=LINE_WIDTH,
            solid_capstyle='round')
    
    # 任務開始的垂直線
    ax.axvline(start_epoch_display, color=colors[g_idx], linestyle=':', alpha=0.7, linewidth=1.5)

# 圖表設定
ax.set_xlim(0.5, global_max_epoch + 0.5)
ax.set_ylim(0, 100)
ax.set_yticks(np.arange(0, 101, 10))
ax.set_yticks(np.arange(0, 101, 5), minor=True)
ax.set_ylabel("Validation Accuracy (%)", fontsize=FONT_SIZE_LABEL)

xtick_step = max(10, round(global_max_epoch / 20 / 10) * 10)
ax.set_xticks(np.arange(1, global_max_epoch + 1, xtick_step))
ax.set_xlabel("Epoch", fontsize=FONT_SIZE_LABEL)

ax.grid(True, linestyle='--', alpha=0.5, which='both')
ax.tick_params(axis='both', labelsize=FONT_SIZE_LABEL - 2)

ax.legend(title="Task (start when introduced)",
          loc='lower left',
          bbox_to_anchor=(0.02, -0.28),
          ncol=4,
          fontsize=FONT_SIZE_LEGEND)

ax.set_title(f"{model} – {method}\n"
             "Task-specific Validation Accuracy over Time\n"
             "(each curve begins at the epoch when the task is first trained, fully connected)",
             fontsize=FONT_SIZE_TITLE + 2, pad=25)

plt.tight_layout(rect=[0, 0.12, 1, 0.96])

# 儲存
filename = f"{model}_{method.replace(' ', '_')}"
plt.savefig(f"{filename}.png", dpi=400, bbox_inches='tight')
#plt.savefig(f"{filename}.pdf", bbox_inches='tight')
plt.show()

print(f"\n圖已儲存為：{filename}.png / .pdf")