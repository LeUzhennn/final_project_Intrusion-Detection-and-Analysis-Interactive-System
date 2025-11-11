# **入侵偵測互動式分析系統 - 專案規格**

## **1. 專案概述 (Project Overview)**

本專案旨在建立一個互動式網頁應用，用於分析網路入侵偵測資料集。使用者將能夠透過網頁介面，進行資料探索、特徵篩選、模型訓練，並即時查看模型效能。此系統的核心是讓非技術背景的使用者也能理解並參與機器學習模型的分析過程。

## **2. 技術堆疊 (Tech Stack)**

*   **程式語言:** Python 3.9+
*   **Web 框架:** Streamlit
*   **資料處理與分析:**
    *   Pandas: 用於資料讀取、清洗和操作。
    *   Numpy: 用於高效的數值計算。
*   **機器學習:**
    *   Scikit-learn: 用於資料預處理 (StandardScaler)、模型建立 (RandomForestClassifier)、資料分割 (train_test_split) 和模型評估 (accuracy_score, confusion_matrix, etc.)。
    *   XGBoost / LightGBM (可選): 作為進階的集成學習模型選項。
*   **資料視覺化:**
    *   Matplotlib / Seaborn: 用於繪製靜態圖表，如混淆矩陣、特徵重要性圖。
    *   Streamlit 內建圖表: 用於快速生成互動式圖表。

## **3. 專案結構 (Project Structure)**

```
.
├── 📄 .gitignore
├── 📄 AGENTS.md
├── 📄 CRISP-DM 分析方向.pdf
├── 📁 data/
│   └── 📄 03-01-2018.csv  // 資料集存放處
├── 📁 openspec/
│   └── 📄 project.md
├── 📄 requirements.txt      // 專案依賴套件
├── 📁 src/
│   ├── 📄 data_loader.py    // 資料讀取與初步清理模組
│   ├── 📄 feature_selector.py // 特徵選擇模組
│   └── 📄 model_trainer.py  // 模型訓練與評估模組
└── 📄 app.py                // Streamlit 應用程式主檔案
```

## **4. 程式碼慣例 (Code Conventions)**

*   **風格指南:** 遵循 [PEP 8 -- Style Guide for Python Code](https://peps.python.org/pep-0008/)。
*   **命名規則:**
    *   變數與函式：使用 `snake_case` (例如: `load_data`)。
    *   類別：使用 `PascalCase` (例如: `ModelTrainer`)。
*   **註解:** 為複雜的邏輯區塊或函式編寫清晰的註解，解釋其 *目的* 而非 *過程*。
*   **型別提示 (Type Hinting):** 盡可能為函式參數和回傳值加上型別提示，以增加程式碼的可讀性和健壯性。
*   **文件字串 (Docstrings):** 為所有公開的模組和函式編寫 Docstrings，說明其功能、參數和回傳值。

## **5. 核心實作計畫 (CRISP-DM)**

### **第一階段：商業理解 (Business Understanding)**

1.  **目標定義：**
    *   **主要目標：** 偵測網路流量中的惡意活動（入侵）。
    *   **專案產出：** 一個 Streamlit 互動網頁，整合資料分析、特徵選擇、模型訓練與評估。
    *   **成功標準：** 能夠載入資料、呈現視覺化圖表、讓使用者選擇特徵、訓練至少一種集成學習模型，並展示如準確率、混淆矩陣等評估指標。

### **第二階段：資料理解 (Data Understanding)**

1.  **資料獲取：**
    *   從 Kaggle 下載 `03-01-2018.csv` 資料集。
2.  **初步資料探索 (EDA - Exploratory Data Analysis)：**
    *   使用 `pandas` 函式庫載入 CSV 檔案。
    *   檢視資料的基本資訊：`info()` 查看各欄位的資料型態與非空值數量。
    *   檢視資料的統計摘要：`describe()` 查看數值特徵的分佈情況。
    *   分析目標變數 `Label`：計算不同標籤的數量，了解資料是否平衡。
    *   視覺化分析：繪製長條圖來顯示 `Label` 的分佈。

### **第三階段：資料準備 (Data Preparation)**

1.  **資料清理：**
    *   處理遺失值與無窮值。
2.  **特徵工程與轉換：**
    *   **編碼 (Encoding)：** 將分類特徵轉換為數值格式。
    *   **資料標準化/歸一化 (Scaling)：** 對特徵進行縮放。
3.  **特徵選擇 (Feature Selection)：**
    *   **過濾法 (Filter Method)：** 計算相關係數。
    *   **嵌入法 (Embedded Method)：** 利用模型的特徵重要性進行篩選。

### **第四階段：模型建立 (Modeling)**

1.  **資料分割：**
    *   將資料集分割為訓練集與測試集。
2.  **模型選擇與訓練：**
    *   專注於**集成學習 (Ensemble Learning)** 模型 (e.g., RandomForest, XGBoost)。

### **第五階段：模型評估 (Evaluation)**

1.  **效能評估：**
    *   使用測試集評估模型，計算 Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC/AUC。

### **第六階段：系統部署 (Deployment)**

1.  **Streamlit 網頁設計：**
    *   **佈局：** 使用側邊欄放置控制項，主頁面顯示結果。
    *   **互動功能：** 允許使用者選擇特徵、模型並觸發訓練與評估。
    *   **結果呈現：** 視覺化呈現評估指標與圖表。
    *   **即時預測功能：** 允許使用者手動輸入特徵值進行單筆預測。

2.  **批次預測功能 (Batch Prediction Feature):**
    *   **目標：** 允許使用者上傳 CSV 檔案，對多筆資料進行批次入侵偵測預測。
    *   **介面：**
        *   在「即時預測」區塊下方新增檔案上傳器 (`st.file_uploader`) 和「執行批次預測」按鈕。
        *   **提供下載範例：** 提供一個按鈕，讓使用者下載包含所有選定特徵欄位的空白 CSV 範例檔案。
        *   **互動式欄位映射：** 上傳檔案後，提供介面讓使用者將上傳檔案的欄位手動映射到模型所需的選定特徵。
    *   **資料處理：**
        *   讀取上傳的 CSV 檔案。
        *   **根據使用者映射處理欄位：** 根據使用者設定的映射關係，提取、轉換並填充模型所需的特徵。
        *   對上傳資料進行與訓練資料相同的預處理（數值轉換、使用已儲存的 `StandardScaler` 進行縮放）。
    *   **預測與結果：**
        *   使用已訓練模型進行批次預測。
        *   將預測結果（解碼後的標籤）新增為新欄位到原始上傳資料中。
        *   顯示預測結果的摘要（例如，各類別預測數量）。
        *   顯示包含預測結果的資料表。
