#!/usr/bin/env python3
"""
將每個 CSV 檔的 Open/High/Low/Close/Volume 欄位
以「第一日值 = 100」來標準化，結果寫入 data_processed 資料夾。

使用：
    python normalize_to_day1.py

假設：
- 原始 CSV 檔放在 ./data 目錄，輸出到 ./data_processed。
- CSV 有日期欄（通常叫 Date 或 index），並包含 Open/High/Low/Close/Volume（不區分大小寫）。
"""

import os
import sys
import glob
import pandas as pd

# 可配置區域
INPUT_DIR = "data"
OUTPUT_DIR = "data_processed"
CSV_GLOB = "*.csv"  # 若需要支援 .CSV 或其他，這裡可以改成 "*.[cC][sS][vV]"

# 欄位首選名稱（會做不區分大小寫的比對）
TARGET_COLS = ["Open", "High", "Low", "Close", "Volume"]

def find_case_insensitive_cols(df, target_cols):
    """
    回傳一個 dict，把標準欄位名對應到 data frame 中實際欄位名（若存在）。
    例如: {"Open": "open", "Close": "Close"}，不存在的欄位不會出現在 dict 裡。
    """
    cols_lower_map = {c.lower(): c for c in df.columns}
    mapping = {}
    for t in target_cols:
        key = t.lower()
        if key in cols_lower_map:
            mapping[t] = cols_lower_map[key]
    return mapping

def normalize_df_day1(df, col_mapping):
    """
    對 df 中的實際欄位（由 col_mapping 提供）做第一日基數化（第一列數值 -> 100）。
    回傳新的 DataFrame（複本）。若第一日為 0 或 NaN，該欄將被設定為 NaN，並會在呼叫處產生警告。
    """
    out = df.copy()
    for std_col, actual_col in col_mapping.items():
        series = out[actual_col].astype(float)  # 轉 float，若有 NaN 會保留
        if series.size == 0:
            print(f"警告: 欄位 {actual_col} 無資料，跳過。")
            continue
        first_val = series.iloc[0]
        if pd.isna(first_val):
            print(f"警告: 檔案 '{current_file}' 欄位 '{actual_col}' 第一日為 NaN，該欄結果將為 NaN。")
            out[actual_col] = pd.NA
            continue
        try:
            if float(first_val) == 0.0:
                print(f"警告: 檔案 '{current_file}' 欄位 '{actual_col}' 第一日為 0，除以零被避開，該欄結果將為 NaN。")
                out[actual_col] = pd.NA
                continue
        except Exception:
            print(f"警告: 無法將欄位 '{actual_col}' 第一日值轉為數字 (value={first_val})，該欄將被跳過。")
            out[actual_col] = pd.NA
            continue

        out[actual_col] = (series / float(first_val)) * 100.0
    return out

if __name__ == "__main__":
    # 檢查資料夾
    if not os.path.isdir(INPUT_DIR):
        print(f"錯誤: 找不到輸入資料夾 '{INPUT_DIR}'。請確認你在正確目錄且資料夾存在。")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pattern = os.path.join(INPUT_DIR, CSV_GLOB)
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"在 '{INPUT_DIR}' 未找到符合模式 {CSV_GLOB} 的 CSV 檔。")
        sys.exit(1)

    print(f"找到 {len(files)} 個檔案，開始處理...")

    # 使用全域變數 current_file 以便在函式中能打印檔名（簡單做法）
    for filepath in files:
        current_file = os.path.basename(filepath)
        try:
            # 讀取 CSV，嘗試解析日期欄
            df = pd.read_csv(filepath, parse_dates=True, infer_datetime_format=True)
        except Exception as e:
            print(f"錯誤: 讀取檔案 '{current_file}' 失敗: {e}. 跳過此檔.")
            continue

        if df.empty:
            print(f"警告: '{current_file}' 為空檔案，跳過。")
            continue

        # 找出欄位對應
        col_map = find_case_insensitive_cols(df, TARGET_COLS)
        if not col_map:
            print(f"警告: '{current_file}' 未找到 Open/High/Low/Close/Volume 欄位中的任何一個，原檔複製到輸出資料夾。")
            out_df = df
        else:
            out_df = normalize_df_day1(df, col_map)

        # 若原檔只有 index 為日期（沒有 Date 欄），嘗試保留行索引不變；否則保留原 DataFrame 結構
        out_path = os.path.join(OUTPUT_DIR, current_file)
        try:
            out_df.to_csv(out_path, index=False)
            print(f"已處理並寫入：{out_path}")
        except Exception as e:
            print(f"錯誤: 無法寫入 '{out_path}': {e}")

    print("全部處理完成。")