"""Create train/val/test splits with stratification for the processed dataset."""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

processed_dir = r"D:\朱江龙\pctps-improved\data\processed"
splits_dir = r"D:\朱江龙\pctps-improved\data\splits"
os.makedirs(splits_dir, exist_ok=True)

df = pd.read_csv(os.path.join(processed_dir, "labels.csv"))
print(f"Loaded {len(df)} proteins")

# Use binary label for stratification
labels = df['binary_label'].values
indices = np.arange(len(df))

# First split: 80% train, 20% temp
train_idx, temp_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=labels
)
temp_labels = labels[temp_idx]

# Second split: 10% val, 10% test (half of temp)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, random_state=42, stratify=temp_labels
)

# Write split files
for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
    path = os.path.join(splits_dir, f"{name}.txt")
    with open(path, "w") as f:
        for i in idx:
            f.write(f"{df.iloc[i]['id']}\n")

# Report stats
for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
    bin_labels = labels[idx]
    ter_labels = df['ternary_label'].values[idx]
    pos = (bin_labels == 1).sum()
    neg = (bin_labels == 0).sum()
    ps_self = (ter_labels == 1).sum()
    ps_part = (ter_labels == 2).sum()
    print(f"{name}: {len(idx)} total | PS={pos}, non-PS={neg} | Self={ps_self}, Part={ps_part}")

print(f"\nSplits saved to: {splits_dir}")
