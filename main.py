"""
主程式 - 低成本空氣品質感測器校正模型
執行整個資料処理、模型訓練、評估和視覺化流程
"""

import sys
import warnings
warnings.filterwarnings('ignore')

# 添加 src 目錄到 Python 路徑
sys.path.insert(0, str(__file__).replace('main.py', 'src'))

import pandas as pd
import numpy as np

# 導入各個模組
from config import (
    TARGET_COL, FEATURE_COLS, SENSOR_FEATURES, WEATHER_FEATURES, 
    TIME_FEATURES, SENSOR_TO_MODEL, TRAIN_TEST_SPLIT_RATIO, TABLES_DIR
)
from data_loader import load_data, display_data_info
from preprocessing import preprocess_data, display_missing_statistics
from feature_engineering import feature_engineering
from models import train_models, predict
from evaluation import evaluate_models, find_best_model, display_comparison
from visualization import plot_all_results
from utils import (
    ensure_output_directories, save_csv, save_model, print_summary_report,
    split_data_by_time, prepare_calibration_results, save_calibration_results
)


def main():
    """
    主函數 - 執行完整的感測器校正流程。
    """
    
    print("\n" + "="*60)
    print("低成本空氣品質感測器校正模型")
    print("="*60)
    print(f"目標污染物: {TARGET_COL}")
    print("="*60)
    
    # ========== 1. 確保輸出目錄 ==========
    ensure_output_directories()
    
    # ========== 2. 載入資料 ==========
    print("\n[步驟 1/7] 載入資料...")
    df = load_data()
    display_data_info(df)
    
    # ========== 3. 資料格式確認 ==========
    print("\n[步驟 2/7] 資料格式確認...")
    print("CSV 已由 load_data() 以分號與逗號小數格式讀取，保留 Date/Time 供後續特徵工程使用。")
    
    # ========== 4. 特徵工程（必須在前處理之前，以創建時間特徵） ==========
    print("\n[步驟 3/7] 特徵工程...")
    df = feature_engineering(df)
    
    # ========== 5. 資料前處理 ==========
    print("\n[步驟 4/7] 資料前處理...")
    df = preprocess_data(df, target_col=TARGET_COL)
    
    # 儲存缺失值統計
    missing_df = display_missing_statistics(df)
    save_csv(missing_df, f"missing_values_statistics_{TARGET_COL}.csv")
    
    # ========== 5. 資料切分 ==========
    print("\n[步驟 5/7] 時間切分資料...")
    
    # 移除不需要的欄位（只保留我們需要的欄位）
    keep_cols = [TARGET_COL] + FEATURE_COLS + ['datetime', 'Date', 'Time']
    df = df[[col for col in keep_cols if col in df.columns]]
    
    # 確保沒有 NaN 在特徵中
    print("移除包含 NaN 的資料列...")
    initial_len = len(df)
    df = df.dropna()
    removed = initial_len - len(df)
    print(f"移除 {removed} 筆包含 NaN 的資料")
    
    df_train, df_test = split_data_by_time(df, test_ratio=1-TRAIN_TEST_SPLIT_RATIO)
    
    # 提取特徵和目標
    X_train = df_train[FEATURE_COLS]
    y_train = df_train[TARGET_COL]
    X_test = df_test[FEATURE_COLS]
    y_test = df_test[TARGET_COL]
    
    print(f"\n特徵數量: {len(FEATURE_COLS)}")
    print(f"特徵: {FEATURE_COLS}")
    
    # ========== 7. 模型訓練 ==========
    print("\n[步驟 6/7] 模型訓練...")
    models = train_models(X_train, y_train)
    
    # ========== 8. 模型評估 ==========
    print("\n[步驟 7/7] 模型評估與比較...")
    evaluation_results, evaluation_df, predictions_dict = evaluate_models(
        models, X_train, y_train, X_test, y_test, TARGET_COL
    )
    
    # 儲存評估結果
    save_csv(evaluation_df, f"model_comparison_{TARGET_COL}.csv")
    
    # 找出最佳模型
    best_model_name, best_result = find_best_model(evaluation_df)
    display_comparison(evaluation_df)
    
    # ========== 9. 視覺化 ==========
    print("\n[進階] 視覺化結果...")
    plot_all_results(
        models, X_train, y_train, X_test, y_test,
        evaluation_df, TARGET_COL, best_model_name, df_test=df_test
    )
    
    # ========== 10. 感測器校正應用 ==========
    print("\n[進階] 準備感測器校正結果...")
    best_model = models[best_model_name]
    y_pred = best_model.predict(X_test)
    
    # 準備校正結果表
    calibration_results = prepare_calibration_results(
        df_test, y_test, y_pred, TARGET_COL, SENSOR_TO_MODEL
    )
    
    # 儲存校正結果
    save_calibration_results(calibration_results, TARGET_COL)
    
    # 顯示校正結果摘要
    print("\n" + "="*60)
    print(f"感測器校正結果摘要 ({TARGET_COL})")
    print("="*60)
    print(f"平均校正前誤差 (MAE): {np.abs(df_test[SENSOR_TO_MODEL.get(TARGET_COL, '')].values - y_test.values).mean():.4f}" 
          if TARGET_COL in SENSOR_TO_MODEL and SENSOR_TO_MODEL[TARGET_COL] in df_test.columns else "N/A")
    print(f"平均校正後誤差 (MAE): {calibration_results['absolute_error'].mean():.4f}")
    print(f"最大校正誤差: {calibration_results['absolute_error'].max():.4f}")
    print(f"最小校正誤差: {calibration_results['absolute_error'].min():.4f}")
    
    # ========== 11. 儲存最佳模型 ==========
    print(f"\n儲存最佳模型...")
    save_model(best_model, f"best_model_{TARGET_COL}.pkl")
    
    # ========== 12. 最終摘要 ==========
    print_summary_report(TARGET_COL, best_model_name, best_result)
    
    print("\n✓ 所有流程完成！")
    print(f"✓ 輸出檔案已儲存至: {TABLES_DIR.parent}")
    
    return models, evaluation_df, best_model_name


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ 執行出錯: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
