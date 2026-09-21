"""
設定檔案 - 儲存所有常數、路徑和模型參數
"""

import os
from pathlib import Path


# ============= 資料處理設定 =============
# 目標污染物變數（可自由切換）
TARGET_COL = "C6H6(GT)"

# 所有可用的污染物目標
AVAILABLE_TARGETS = ["CO(GT)", "NOx(GT)", "NO2(GT)", "C6H6(GT)"]

# 感測器反應值特徵（已移除 PT08.S2(NMHC)，因為 NMHC(GT) 缺失值過高 88%）
SENSOR_FEATURES = [
    "PT08.S1(CO)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)"
]

# 氣象變數特徵
WEATHER_FEATURES = ["T", "RH", "AH"]

# 時間特徵
TIME_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "season",
    "rush_hour"
]

# 完整特徵集合
FEATURE_COLS = SENSOR_FEATURES + WEATHER_FEATURES + TIME_FEATURES

# 缺失值代碼
MISSING_VALUE_CODE = -200

# 時間切分比例
TRAIN_TEST_SPLIT_RATIO = 0.8  # 前 80% 訓練，後 20% 測試

# ============= 路徑設定 =============
# 專案根目錄
PROJECT_ROOT = Path(__file__).parent.parent

# 資料目錄
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "AirQualityUCI.csv"
DATA_FILE_XLSX = DATA_DIR / "AirQualityUCI.xlsx"

# 輸出目錄（每個 target 存放到不同子資料夾，避免覆寫）
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# 安全化 TARGET_COL 為資料夾名稱（移除或替換特殊字元）
def _safe_target_name(target: str) -> str:
    name = str(target)
    for ch in ['(', ')', '/', '\\', ' ', '%']:
        name = name.replace(ch, '_')
    # 減少連續下劃線
    while '__' in name:
        name = name.replace('__', '_')
    return name.strip('_')

TARGET_SAFE = _safe_target_name(TARGET_COL)

FIGURES_DIR = OUTPUT_DIR / "figures" / TARGET_SAFE
TABLES_DIR = OUTPUT_DIR / "tables" / TARGET_SAFE
MODELS_DIR = OUTPUT_DIR / "models" / TARGET_SAFE

# 確保輸出目錄存在（會建立 target-specific 的子資料夾）
for directory in [FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)



# ============= 模型參數設定 =============
RANDOM_STATE = 42

# Random Forest 參數
RF_PARAMS = {
    "n_estimators": 300,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

# XGBoost 參數
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 5,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

# ============= 模型名稱對應 =============
SENSOR_TO_MODEL = {
    "CO(GT)": "PT08.S1(CO)",
    "NOx(GT)": "PT08.S3(NOx)",
    "NO2(GT)": "PT08.S4(NO2)",
    "C6H6(GT)": "PT08.S2(NMHC)"
}

# ============= 其他設定 =============
RANDOM_STATE = 42
TEST_SIZE = 0.2

print(f"專案根目錄: {PROJECT_ROOT}")
print(f"資料目錄: {DATA_DIR}")
print(f"輸出目錄: {OUTPUT_DIR}")
