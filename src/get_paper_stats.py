
import numpy as np
import os
import sys

def get_counts(y):
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts))

def print_stats():
    base_dir = r"e:\The\data\processed_rr_history" # Absolute path
    splits = ['train', 'val', 'test']
    classes = [0, 1, 2, 3, 4]
    class_names = ['N', 'S', 'V', 'F', 'Q']

    print("| Class | Train | Val | Test | Total |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    # Store totals
    grand_totals = {c: 0 for c in classes}
    
    # Load data
    data = {}
    for split in splits:
        try:
            y = np.load(os.path.join(base_dir, f"y_{split}.npy"))
            data[split] = get_counts(y)
        except Exception as e:
            print(f"Error loading {split}: {e}")
            data[split] = {}

    # Print rows
    for i, c in enumerate(classes):
        row = f"| **{class_names[i]}** |"
        row_total = 0
        for split in splits:
            count = data[split].get(c, 0)
            row += f" {count:,} |"
            row_total += count
            grand_totals[c] += count
        row += f" **{row_total:,}** |"
        print(row)
        
    # Print Total Row
    row = "| **Total** |"
    for split in splits:
        total_split = sum(data[split].get(c, 0) for c in classes)
        row += f" **{total_split:,}** |"
    row += f" **{sum(grand_totals.values()):,}** |"
    print(row)

if __name__ == "__main__":
    print_stats()
