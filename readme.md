# GPT-5.0 Pure Price Stock Picking  
**純價格資料讓 GPT-5.0 選股實驗**  

**實驗核心**：只給 ChatGPT-5.0 股票的歷史價格資料（OHLCV，無基本面、財報、消息面），它能選出好股嗎？

## 專案流程

1. **資料下載**  
   - `save_sp500_prices.py`：從 Yahoo Finance 下載 S&P 500 成分股最近一年的日線價格資料（開高低收 + 成交量）。  
   - 輸出 → `data/` 資料夾（約 504 個 CSV，每檔股票一個）。

2. **資料正規化**  
   - `normalize_to_day1.py`：以每檔股票的第一天收盤價為基準（設為 100），計算相對價格走勢。  
   - 輸出 → `data_processed/` 資料夾（方便比較不同股票的相對強弱）。

3. **匿名化 + 拆分**  
   - `rename_and_merge_csvs.py`：  
     - 隨機給每檔股票一個 5 字母英文代號（避免 GPT 認出真實公司名）。  
     - 映射表記錄在 `mapping.txt`。  
     - 把所有資料平均拆成 50 個 CSV，放到 `input/` 資料夾（方便分批餵給 GPT，避免 token 限制）。

4. **GPT 選股**  
   - `pick_top10.py`：**由 ChatGPT-5.0 直接生成** 的選股邏輯腳本。  
   - 從 input/ 的匿名資料中，選出它認為的 top 10 強勢股。  
   - 輸出 → `output/`（目前可放結果或 log）。

5. **投資組合優化（額外模組）**  
   - `Monte-Carlo-Portfolio-Optimization-main.py`：使用蒙地卡羅模擬（Monte Carlo Simulation）找出資產配置的最佳權重，最大化報酬 / 最小化風險（基於現代投資組合理論 Efficient Frontier）。  
   - **重要感謝與引用**：  
     這個模組來自 Hunter Gould 的開源專案，我僅借用並整合進本實驗。  
     原作者：**Hunter Gould**  
     原 repo：https://github.com/Gouldh/Monte-Carlo-Portfolio-Optimization  
     原建立日期：2023 年 10 月  

## 貢獻者與感謝

- **ChatGPT-5.0**：撰寫了專案中 99% 的 Python 程式碼
- **Yahoo Finance**：免費提供高品質歷史價格資料。
- **Hunter Gould**：Monte-Carlo-Portfolio-Optimization 模組原作者。
- **小言**：專案構想發起人。

## 如何運行

1. Clone repo：
   ```bash
   git clone https://github.com/danyoko-bugover/GPT5.0-pureprice-stockpicking.git
   cd GPT5.0-pureprice-stockpicking