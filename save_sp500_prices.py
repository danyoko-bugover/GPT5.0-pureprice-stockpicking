#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
下載 S&P 500 成分股近一年每日價格，並儲存為 CSV。
- 來源成分股清單：Wikipedia (List of S&P 500 companies)
- 價格資料來源：yfinance (Yahoo Finance)
- 每檔個別儲存為 ./data/{TICKER}.csv
- 也會建立 ./data/all_sp500_last_year.csv (含 ticker 欄)

注意：
- 這個腳本採取較保守的請求頻率以避免被封鎖。
- 如需加速，可移除 time.sleep 或使用平行下載 (需小心處理 rate limit)。
"""

import os
import time
from datetime import datetime, timedelta
import argparse

import requests
import pandas as pd
from bs4 import BeautifulSoup
import yfinance as yf

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DATA_DIR = "data"
SLEEP_BETWEEN_TICKERS = 1.0  # seconds, 可視情況調整（越大越慢越保險）

def fetch_sp500_tickers():
    """
    使用 Wikipedia MediaWiki API 取得 S&P 500 成分股表格並回傳 ticker list。
    回傳值：list of str (tickers)，例如 ['AAPL', 'MSFT', 'BRK-B', ...]
    """
    API_URL = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "page": "List of S&P 500 companies",  # 使用未編碼的標題
        "prop": "text",
        "format": "json",
        "formatversion": 2,
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; save_sp500_prices/1.0; +https://example.com/)",
        "Accept-Language": "en-US,en;q=0.9",
    }

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(API_URL, params=params, headers=headers, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            if "parse" not in data or "text" not in data["parse"]:
                raise RuntimeError("Wikipedia API 回傳格式不正確，找不到 parse.text。")

            html = data["parse"]["text"]

            # 使用 pandas 解析 HTML 表格
            dfs = pd.read_html(html)
            if not dfs:
                raise RuntimeError("從 Wikipedia API 解析的 HTML 中找不到任何表格。")

            # 找到包含 symbol/ticker 欄位的表格（通常是第一個符合的）
            table = None
            for df in dfs:
                cols = [str(c).lower() for c in df.columns]
                if any("symbol" in c or "ticker" in c for c in cols):
                    table = df
                    break
            if table is None:
                # fallback: 使用第一個表格
                table = dfs[0]

            # 嘗試找到 symbol 欄位名稱
            col_candidates = [c for c in table.columns if "symbol" in str(c).lower() or "ticker" in str(c).lower()]
            if col_candidates:
                raw_tickers = table[col_candidates[0]].astype(str).tolist()
            else:
                raw_tickers = table.iloc[:, 0].astype(str).tolist()

            # 清理 ticker（移除空白、將 '.' 轉為 '-'，忽略空項）
            cleaned = []
            for t in raw_tickers:
                if t is None:
                    continue
                t = str(t).strip()
                if not t:
                    continue
                # 有些條目可能包含註解或額外文字，僅取第一段（例如 "BRK.B Berkshire Hathaway"）
                # 以空白或 newline 切割取第一部分，並移除逗號等
                t = t.split()[0].strip().strip(',')
                # Yahoo/Yfinance 使用 '-' 取代 '.'
                t = t.replace('.', '-')
                cleaned.append(t)
            if not cleaned:
                raise RuntimeError("解析後沒有找到任何 ticker。")
            return cleaned

        except requests.HTTPError as e:
            if attempt < max_attempts:
                wait = 5 * attempt
                print(f"[WARN] Wikipedia API request failed (attempt {attempt}/{max_attempts}): {e}. 等待 {wait}s 後重試...")
                time.sleep(wait)
                continue
            else:
                raise
        except ValueError as e:
            # JSON decode 等錯誤
            raise RuntimeError(f"Wikipedia API 回應無法解析為 JSON: {e}")
        except Exception:
            # 仍拋出錯誤讓上層捕捉並顯示
            raise

    # 簡單重試機制
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(API_URL, params=params, headers=headers, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            if "parse" not in data or "text" not in data["parse"]:
                raise RuntimeError("Wikipedia API 回傳格式不正確，找不到 parse.text。")
            html = data["parse"]["text"]
            # 使用 pandas 直接解析 HTML 表格（通常第一個 wikitable 是我們要的）
            dfs = pd.read_html(html)
            if not dfs:
                raise RuntimeError("從 Wikipedia API 解析出來的 HTML 中找不到任何表格。")
            # 找出包含 Symbol 欄位的表格（較安全）
            table = None
            for df in dfs:
                cols = [str(c).lower() for c in df.columns]
                if any("symbol" in c or "ticker" in c for c in cols):
                    table = df
                    break
            if table is None:
                # fallback: 使用第一個表格
                table = dfs[0]

            # 嘗試找出 symbol 欄位名稱
            col_candidates = [c for c in table.columns if "Symbol" in str(c) or "Ticker" in str(c) or "symbol" in str(c).lower() or "ticker" in str(c).lower()]
            if col_candidates:
                tickers = table[col_candidates[0]].astype(str).tolist()
            else:
                # 萬一沒找到，取第一欄
                tickers = table.iloc[:, 0].astype(str).tolist()

            # 清理 tickers（移除空白，將 '.' 轉為 '-' 以符合 Yahoo 的代號格式）
            cleaned = []
            for t in tickers:
                t = t.strip()
                if not t:
                    continue
                t = t.replace(".", "-")
                cleaned.append(t)
            return cleaned

        except requests.HTTPError as e:
            # 若是 4xx/5xx，對 403 可稍等再試
            if attempt < max_attempts:
                wait = 5 * attempt
                print(f"[WARN] Wikipedia API request failed (attempt {attempt}/{max_attempts}): {e}. 等待 {wait}s 後重試...")
                time.sleep(wait)
                continue
            else:
                raise
        except Exception:
            # 其他解析或 JSON 錯誤
            raise

def download_price_for_ticker(ticker, period_days=365):
    """
    使用 yfinance 下載 ticker 的過去 period_days 的每日資料。
    回傳 pandas DataFrame（index 為 Date），包含 Open, High, Low, Close, Adj Close, Volume
    """
    end = datetime.utcnow().date()
    start = end - timedelta(days=period_days)
    # yfinance 的 period 或 start/end 都可以用，我用 start/end
    try:
        # 使用 Ticker.history
        tk = yf.Ticker(ticker)
        df = tk.history(start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(), interval="1d", auto_adjust=False)
        # 若 df 為空，嘗試用 yf.download 作為 fallback
        if df is None or df.empty:
            df = yf.download(ticker, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(), interval="1d", progress=False)
        if df is None:
            return None
        # 確保 index 為日期（yyyy-mm-dd）
        df.index = pd.to_datetime(df.index).date
        # 重命名 'Adj Close' 欄為 'Adj Close'（若 yfinance 為 'Adj Close' 則保留）
        # 保留原本欄位 Open, High, Low, Close, Adj Close, Volume
        cols_expected = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        # 若 'Adj Close' 不存在但有 'Adj Close' 的小差異名稱，處理它
        if "Adj Close" not in df.columns and "Adj Close" in df.columns:
            df = df.rename(columns={df.columns[df.columns.str.contains("Adj", case=False)][0]: "Adj Close"})
        # 過濾欄位
        available = [c for c in cols_expected if c in df.columns]
        df = df[available]
        return df
    except Exception as e:
        print(f"[ERROR] 下載 {ticker} 時發生錯誤: {e}")
        return None

def main(output_dir=DATA_DIR, sleep_sec=SLEEP_BETWEEN_TICKERS, period_days=365, save_concatenated=True):
    os.makedirs(output_dir, exist_ok=True)
    print("取得 S&P 500 成分股清單...")
    tickers = fetch_sp500_tickers()
    print(f"找到 {len(tickers)} 檔成分股，開始逐一下載（每檔暫停 {sleep_sec} 秒）...")
    all_rows = []
    failed = []
    for i, t in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] 下載 {t} ...")
        df = download_price_for_ticker(t, period_days=period_days)
        if df is None or df.empty:
            print(f"  -> 無資料或下載失敗，跳過 {t}")
            failed.append(t)
        else:
            # 儲存個別 CSV
            # 在 CSV 中，加入一個 Date 欄
            df_to_save = df.copy()
            df_to_save.insert(0, "Date", pd.to_datetime(df_to_save.index))
            csv_path = os.path.join(output_dir, f"{t}.csv")
            df_to_save.to_csv(csv_path, index=False)
            print(f"  -> 儲存為 {csv_path}（{len(df_to_save)} 列）")
            # 準備加入合併表（以標準化的欄位）
            concat_row = df_to_save.copy()
            concat_row.insert(0, "Ticker", t)
            all_rows.append(concat_row)
        time.sleep(sleep_sec)

    # 合併成一個 CSV
    if save_concatenated and all_rows:
        print("合併所有股票成單一 CSV 檔...")
        combined = pd.concat(all_rows, ignore_index=True, sort=False)
        combined_csv_path = os.path.join(output_dir, "all_sp500_last_year.csv")
        combined.to_csv(combined_csv_path, index=False)
        print(f"合併檔已儲存為 {combined_csv_path}（{len(combined)} 列）")
    if failed:
        print("以下 ticker 下載失敗：")
        print(", ".join(failed))
    print("完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下載 S&P 500 成分股近一年價格並儲存為 CSV")
    parser.add_argument("--out", "-o", help="輸出資料夾 (預設: data)", default=DATA_DIR)
    parser.add_argument("--sleep", "-s", type=float, help="每檔下載之間暫停秒數 (預設: 1.0)", default=SLEEP_BETWEEN_TICKERS)
    parser.add_argument("--days", "-d", type=int, help="下載過去幾天 (預設: 365)", default=365)
    parser.add_argument("--no-concat", dest="concat", action="store_false", help="不要建立合併的 all_sp500_last_year.csv")
    args = parser.parse_args()
    main(output_dir=args.out, sleep_sec=args.sleep, period_days=args.days, save_concatenated=args.concat)