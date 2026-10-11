import pandas as pd
from pathlib import Path

folder_path = Path("./")
print("Calculating actual elapsed time for each Set ...\n")


total_seconds = 0.0
results = []

for set_num in range(1, 8):
    file_name = f"training_metrics_set{set_num}.csv"
    file_path = folder_path / file_name
    
    if not file_path.exists():
        print(f"❌ {file_name} Does not exist")
        continue
    
    df = pd.read_csv(file_path)
    
    if "Training_Duration (s)" not in df.columns:
        print(f"❌ {file_name} is missing the 'Training_Duration (s)' column")
        continue
    
    last_duration = pd.to_numeric(df["Training_Duration (s)"].iloc[-1], errors='coerce')
    
    if pd.isna(last_duration):
        print(f"⚠️ {file_name} ")
        continue
    
    set_hours = last_duration / 3600
    total_seconds += last_duration
    
    results.append({
        "Set": set_num,
        "Epoch": len(df),
        "Training Time(hr)": round(set_hours, 4),
    })
    
    print(f"Set {set_num:2d} → Actual total time: {set_hours:6.2f} hours ({len(df)} epochs)")


# ============================================
if total_seconds > 0:
    total_hours = total_seconds / 3600
    total_days = total_hours / 24
    
    print("\n" + "="*65)
    print(f"✅ Total training time for 7 sets: {total_hours:.2f} hours")
    print(f"                               ({total_days:.2f} days)")
    print("="*65)

    
    summary_df = pd.DataFrame(results)
    print("\nDetailed time elapsed for each Set:")
    print(summary_df.to_string(index=False))
    
    # Save
    summary_df.to_csv("training_time.csv", index=False, encoding="utf-8-sig")
    print(f"\n📄 save training_time.csv")
else:
    print("error")