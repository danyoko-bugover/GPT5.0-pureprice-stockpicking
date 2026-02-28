### 關於專案
讓ChatGPT僅有股票價格訊息，它怎麼選股呢？

### 構成
save_sp500_price.py: 從Yahoo Finance下載標普500指數成份股，最近一年的價格數據。包括開市價、收市價、日內最高價、日內最低價和交易量。資料會存放於 data 資料夾下。
normalize_to_day1.py：以第一天數據為基數100，表示其他日期的價格數據。如第一天的價格為5元；第二天的價格為6元；第三天的價格為4元，則會編輯被第一天：5/5 * 100 = 100；第二天：6/5 * 100 = 120；第三天：4/5 * 100 = 80。資料會存放於 data_processed 資料夾下。
rename_and_merge_csvs.py：給予所有股票一個隨機的五英文字母名稱。mapping.txt會記錄股本原名和隨機名稱。其後會把所有價格數據平均拆分於50個csv檔案內，存放於 input內。
pick_top10.py： **由ChatGPT-5.0編寫** 的選股腳本。
Monte-Carlo-Portfolio-Optimization/main.py：由蒙地卡落法找出資產組合理論最佳的投資比重。

### 貢獻
ChatGPT-5.0-min：編寫所有99%的代碼。
Yahoo Finance：數據提供方。
Hunter Gould：Monte-Carlo-Portfolio-Optimization的編寫者。
小言：提出主意# PurePrice_GPT5.0StockPicking
