import os
import pandas as pd
from collections import defaultdict
import glob

def analyze_dataset(base_path):
    # 1. Inspect TamilChar.csv
    csv_path = os.path.join(base_path, 'TamilChar.csv')
    try:
        df = pd.read_csv(csv_path)
        num_classes = len(df)
        print(f"TamilChar.csv found. Number of classes defined: {num_classes}")
        print("First 5 rows:")
        print(df.head())
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return

    # 2. Inspect train directory
    train_dir = os.path.join(base_path, 'train')
    train_classes = defaultdict(int)
    total_train = 0
    if os.path.exists(train_dir):
        for user_dir in os.listdir(train_dir):
            user_path = os.path.join(train_dir, user_dir)
            if os.path.isdir(user_path):
                for file in os.listdir(user_path):
                    if file.endswith(('.tiff', '.tif', '.png', '.jpg')):
                        try:
                            # e.g., '000t01.tiff' -> class 0
                            class_id = int(file[:3])
                            train_classes[class_id] += 1
                            total_train += 1
                        except ValueError:
                            print(f"Warning: Unexpected filename format in train: {file}")
    print(f"\nTotal train samples: {total_train}")
    print(f"Number of unique classes in train: {len(train_classes)}")

    # 3. Inspect test directory and ground_truth.txt
    test_dir = os.path.join(base_path, 'test')
    gt_path = os.path.join(base_path, 'ground_truth.txt')
    
    test_classes = defaultdict(int)
    total_test_images = 0
    if os.path.exists(test_dir):
        test_images = [f for f in os.listdir(test_dir) if f.endswith(('.tiff', '.tif', '.png', '.jpg'))]
        total_test_images = len(test_images)
        print(f"\nTotal test images in directory: {total_test_images}")
    
    total_gt = 0
    if os.path.exists(gt_path):
        with open(gt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_gt = len(lines)
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        # Format is usually: <filename_prefix> <class_id>
                        class_id = int(parts[1])
                        test_classes[class_id] += 1
                    except ValueError:
                        pass
        print(f"Total entries in ground_truth.txt: {total_gt}")
        print(f"Number of unique classes in test (from GT): {len(test_classes)}")
    else:
        print("ground_truth.txt not found")
        
    print("\nSummary of class distribution (Train):")
    # Print max, min, avg samples per class
    if train_classes:
        counts = list(train_classes.values())
        print(f"  Min samples/class: {min(counts)}")
        print(f"  Max samples/class: {max(counts)}")
        print(f"  Avg samples/class: {sum(counts)/len(counts):.2f}")

if __name__ == "__main__":
    analyze_dataset('data/TamilNet_old')
