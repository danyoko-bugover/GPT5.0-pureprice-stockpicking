#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rename_and_merge_csvs.py

產生五字母亂碼 ticket、平均分成 10 組合併、刪除暫存 renamed_temp、保留 mapping.txt。
此版本會排除清單 EXCLUDE_FILENAMES（預設包含 "all_sp500_last_year.csv"），
排除時不考慮大小寫（lowercased 比對），也接受只寫檔名（含副檔名）。
"""

import os
import sys
import shutil
import random
import string
from pathlib import Path
import pandas as pd

# ---------- 可修改設定 ----------
SRC_DIR = Path("data_processed")
OUT_DIR = Path("input")
TMP_DIR = Path("renamed_temp")
MAPPING_FILE = Path("mapping.txt")
NUM_OUTPUT_FILES = 50
NAME_LENGTH = 5
USE_LOWERCASE = True
RANDOM_SEED = None
TICKET_COLNAME = "ticket"

# 要排除的檔名清單（不區分大小寫），填入檔名（含副檔名）或純檔名
EXCLUDE_FILENAMES = {
    "all_sp500_last_year.csv",
    # 若還要排除其他，請在此加入，例如 "example.csv"
}
# ----------------------------------

def random_name(existing:set, length:int=NAME_LENGTH, lowercase:bool=USE_LOWERCASE):
    letters = string.ascii_lowercase if lowercase else string.ascii_uppercase
    while True:
        name = ''.join(random.choices(letters, k=length))
        if name not in existing:
            return name

def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    if not SRC_DIR.exists() or not SRC_DIR.is_dir():
        print(f"來源資料夾不存在：{SRC_DIR.resolve()}")
        sys.exit(1)

    # 取得所有 .csv 檔（不遞迴），並排除 EXCLUDE_FILENAMES 中的檔案（以小寫比對）
    all_csvs = sorted([p for p in SRC_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])
    filtered = []
    for p in all_csvs:
        if p.name.lower() in {name.lower() for name in EXCLUDE_FILENAMES}:
            print(f"排除檔案（不會處理）：{p.name}")
            continue
        filtered.append(p)

    total = len(filtered)
    if total == 0:
        print(f"來源資料夾在排除名單後沒有任何要處理的 CSV 檔。")
        sys.exit(1)
    print(f"找到 {len(all_csvs)} 個 CSV，排除清單後剩下 {total} 個要處理的 CSV。")

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    used_names = set()
    mapping = []  # list of tuples (new_basename, original_path)

    print("複製並隨機命名檔案到暫存資料夾（排除指定檔案）...")
    for p in filtered:
        new_basename = random_name(used_names)
        used_names.add(new_basename)
        new_filename = new_basename + ".csv"
        dest = TMP_DIR / new_filename

        attempt = 0
        while dest.exists():
            attempt += 1
            new_basename = random_name(used_names)
            used_names.add(new_basename)
            new_filename = new_basename + ".csv"
            dest = TMP_DIR / new_filename
            if attempt > 10000:
                raise RuntimeError("無法產生唯一檔名，請檢查程式邏輯。")

        shutil.copy2(p, dest)
        mapping.append((new_basename, str(p)))
    print(f"完成複製與命名，共產生 {len(mapping)} 個重新命名的檔案。")

    # 分桶
    n = len(mapping)
    per_bucket = n // NUM_OUTPUT_FILES
    remainder = n % NUM_OUTPUT_FILES

    buckets = []
    idx = 0
    for i in range(NUM_OUTPUT_FILES):
        count = per_bucket + (1 if i < remainder else 0)
        bucket = mapping[idx: idx + count]
        buckets.append(bucket)
        idx += count

    print(f"開始合併成 {NUM_OUTPUT_FILES} 個 CSV 檔案到資料夾：{OUT_DIR.resolve()}")
    for i, bucket in enumerate(buckets, start=1):
        if not bucket:
            print(f"Bucket {i} 為空，跳過。")
            continue

        outpath = OUT_DIR / f"merged_part_{i}.csv"
        first_write = True
        for new_basename, original_path in bucket:
            srcfile = TMP_DIR / (new_basename + ".csv")
            try:
                df = pd.read_csv(srcfile)
            except Exception as e:
                print(f"警告：讀取 {srcfile} 失敗，錯誤：{e}。此檔會被跳過。")
                continue

            df[TICKET_COLNAME] = new_basename

            if first_write:
                df.to_csv(outpath, index=False, mode="w", encoding="utf-8")
                first_write = False
            else:
                df.to_csv(outpath, index=False, mode="a", header=False, encoding="utf-8")

        if first_write:
            print(f"Bucket {i} 沒有成功寫入任何資料，已跳過建立 {outpath}.")
        else:
            print(f"輸出：{outpath.resolve()}")

    # 寫 mapping.txt（保留 new_basename\toriginal_path）
    print(f"寫入對照檔：{MAPPING_FILE.resolve()}")
    with open(MAPPING_FILE, "w", encoding="utf-8") as f:
        f.write("# new_basename\toriginal_path\n")
        for new_basename, original_path in mapping:
            f.write(f"{new_basename}\t{original_path}\n")

    # 刪除暫存資料夾 renamed_temp
    print("嘗試刪除暫存資料夾：", TMP_DIR.resolve())
    try:
        shutil.rmtree(TMP_DIR)
        print("已刪除暫存資料夾。")
    except Exception as e:
        print(f"警告：刪除暫存資料夾失敗：{e}。請手動刪除 {TMP_DIR.resolve()}")

    print("全部完成。被排除的檔案仍留在 data_processed（如 all_sp500_last_year.csv）。mapping.txt 已保留，用於對照亂碼與原始檔名。")

if __name__ == "__main__":
    main()