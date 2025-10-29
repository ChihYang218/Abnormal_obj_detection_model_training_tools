#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=============================================================================
count_clss.py (YOLO 標籤計數腳本)
=============================================================================

[用途]
這個腳本會掃描 './dataset' 目錄下的 'train', 'valid', 和 'test' 資料夾。
它會讀取所有位於 'labels/' 子目錄中的 .txt 標籤檔案，並統計每個類別
(class) 的總出現次數。

YOLO 標籤格式假定為：
<class_id> <x_center> <y_center> <width> <height>
腳本會讀取每行的第一個數字 (<class_id>) 來進行計數。

[類別對應]
- 0: bird
- 1: person
- 2: pig

[如何使用]
執行: python3 count_clss.py
"""

import os
from pathlib import Path
from collections import defaultdict

# --- 腳本設定 ---

# 1. 資料集根目錄
DATASET_DIR = Path("../dataset")

# 2. 要掃描的子目錄 (splits)
SPLITS_TO_SCAN = ["train", "valid", "test"]

# 3. 類別 ID 與名稱的對應
CLASS_MAP = {
    0: "bird",
    1: "person",
    2: "pig"
}

# --- 程式主體 ---

def count_yolo_labels():
    """
    遍歷所有 split 的 labels 資料夾，並統計 class ID。
    """
    print(f"🔍 開始掃描 {DATASET_DIR} ...")

    # 建立一個巢狀字典來儲存計數
    # 結構: split_counts['train']['bird'] = 10
    split_counts = {split: defaultdict(int) for split in SPLITS_TO_SCAN}
    
    # 建立一個字典來儲存總計數
    total_counts = defaultdict(int)
    
    total_files_processed = 0
    
    for split in SPLITS_TO_SCAN:
        labels_dir = DATASET_DIR / split / "labels"
        
        if not labels_dir.is_dir():
            print(f"⚠️  警告：找不到 '{labels_dir}' 資料夾，跳過 '{split}'。")
            continue
            
        print(f"  Processing '{split}' split...")
        
        # 找出所有 .txt 檔案
        label_files = list(labels_dir.glob("*.txt"))
        
        if not label_files:
            print(f"  -> 在 '{labels_dir}' 中沒有找到 .txt 檔案。")
            continue
            
        total_files_processed += len(label_files)
        
        for txt_file in label_files:
            try:
                with open(txt_file, 'r') as f:
                    for line in f:
                        # 去除前後空白並用空格分割
                        parts = line.strip().split()
                        
                        if not parts:
                            continue # 跳過空行
                            
                        try:
                            # YOLO 格式中，第一個數字是 class ID
                            class_id = int(parts[0])
                            
                            # 檢查 class_id 是否在我們的對應表中
                            if class_id in CLASS_MAP:
                                class_name = CLASS_MAP[class_id]
                                split_counts[split][class_name] += 1
                                total_counts[class_name] += 1
                            else:
                                # 發現了未定義的 class ID
                                unknown_name = f"unknown_id_{class_id}"
                                split_counts[split][unknown_name] += 1
                                total_counts[unknown_name] += 1
                                
                        except ValueError:
                            # 該行第一個字不是數字
                            print(f"  -> 警告：在 {txt_file} 中發現格式錯誤的行：{line.strip()}")
                            
            except Exception as e:
                print(f"❌ 錯誤：無法讀取檔案 {txt_file}：{e}")
                
    # --- 輸出結果 ---
    
    if total_files_processed == 0:
        print("\n❌ 錯誤：在所有指定的 'labels' 資料夾中都沒有找到任何 .txt 檔案。")
        print("請確認 'dataset' 資料夾結構是否正確。")
        return

    print("\n" + "="*30)
    print("📊 類別統計結果 (依資料集)")
    print("="*30)
    
    for split in SPLITS_TO_SCAN:
        print(f"\n--- {split.upper()} ---")
        counts = split_counts[split]
        if not counts:
            print("  (此資料集中無任何標籤)")
        else:
            # 排序以確保輸出順序一致
            for class_name, count in sorted(counts.items()):
                print(f"  {class_name:<10}: {count} 個")

    print("\n" + "="*30)
    print("📈 類別總計 (Grand Total)")
    print("="*30)
    
    if not total_counts:
         print("  (未統計到任何類別)")
    else:
        grand_total = 0
        # 排序以確保輸出順序一致
        for class_name, count in sorted(total_counts.items()):
            print(f"  {class_name:<10}: {count} 個")
            if not class_name.startswith("unknown"):
                grand_total += count
        print("---")
        print(f"  {'TOTAL (known)':<10}: {grand_total} 個標籤")


if __name__ == "__main__":
    count_yolo_labels()