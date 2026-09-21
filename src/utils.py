"""
工具模組 - 通用工具函數
"""

import os
import pandas as pd
import pickle
from pathlib import Path

from config import TABLES_DIR, MODELS_DIR


def ensure_output_directories():
    """
    確保所有輸出目錄存在。
    """
    
    for directory in [TABLES_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def save_csv(df, filename, output_dir=TABLES_DIR):
    """
    儲存 DataFrame 為 CSV 檔案。
    
    Args:
        df (pd.DataFrame): 要儲存的資料框
        filename (str): 檔案名稱
        output_dir (Path): 輸出目錄，預設為 TABLES_DIR
    
    Returns:
        Path: 儲存的檔案路徑
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / filename
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    
    print(f"✓ 已儲存: {filepath}")
    return filepath


def save_model(model, filename, output_dir=MODELS_DIR):
    """
    儲存模型為 pickle 檔案。
    
    Args:
        model: 要儲存的模型
        filename (str): 檔案名稱
        output_dir (Path): 輸出目錄，預設為 MODELS_DIR
    
    Returns:
        Path: 儲存的檔案路徑
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / filename
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✓ 已儲存模型: {filepath}")
    return filepath


def load_model(filename, output_dir=MODELS_DIR):
    """
    載入 pickle 格式的模型。
    
    Args:
        filename (str): 檔案名稱
        output_dir (Path): 輸出目錄，預設為 MODELS_DIR
    
    Returns:
        model: 載入的模型
    """
    
    filepath = Path(output_dir) / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"模型檔案不存在: {filepath}")
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✓ 已載入模型: {filepath}")
    return model


def prepare_calibration_results(df_test, y_test, y_pred, target_col, sensor_mapping):
    """
    準備感測器校正結果表。
    
    Args:
        df_test (pd.DataFrame): 測試集資料
        y_test (pd.Series): 真實值
        y_pred (array): 預測值
        target_col (str): 目標欄位名稱
        sensor_mapping (dict): 污染物到感測器的對應字典
    
    Returns:
        pd.DataFrame: 校正結果表
    """
    
    # 取得對應的感測器欄位
    sensor_col = sensor_mapping.get(target_col, "")
    
    # 建立結果 DataFrame
    results = pd.DataFrame({
        "datetime": df_test["datetime"],
        "y_true": y_test.values,
        "raw_sensor_value": df_test[sensor_col].values if sensor_col and sensor_col in df_test.columns else np.nan,
        "y_pred_calibrated": y_pred,
    })
    
    # 計算絕對誤差
    results["absolute_error"] = np.abs(results["y_true"] - results["y_pred_calibrated"])
    
    return results


def save_calibration_results(results, target_col, output_dir=TABLES_DIR):
    """
    儲存校正結果 CSV 檔案。
    
    Args:
        results (pd.DataFrame): 校正結果表
        target_col (str): 目標欄位名稱
        output_dir (Path): 輸出目錄
    
    Returns:
        Path: 儲存的檔案路徑
    """
    
    filename = f"calibrated_results_{target_col}.csv"
    return save_csv(results, filename, output_dir)


def print_summary_report(target_col, best_model_name, evaluation_result):
    """
    列印摘要報告。
    
    Args:
        target_col (str): 目標欄位名稱
        best_model_name (str): 最佳模型名稱
        evaluation_result (dict): 評估結果
    """
    
    print("\n" + "="*60)
    print("感測器校正模型 - 最終摘要報告")
    print("="*60)
    
    print(f"\n目標污染物: {target_col}")
    print(f"最佳模型: {best_model_name}")
    
    print(f"\n訓練集表現:")
    print(f"  - MAE: {evaluation_result['train_MAE']:.4f}")
    print(f"  - RMSE: {evaluation_result['train_RMSE']:.4f}")
    print(f"  - R2: {evaluation_result['train_R2']:.4f}")
    
    print(f"\n測試集表現:")
    print(f"  - MAE: {evaluation_result['test_MAE']:.4f}")
    print(f"  - RMSE: {evaluation_result['test_RMSE']:.4f}")
    print(f"  - R2: {evaluation_result['test_R2']:.4f}")
    
    print("\n" + "="*60)


def split_data_by_time(df, test_ratio=0.2):
    """
    使用時間切分法分割資料。
    
    前 (1-test_ratio)% 為訓練集，後 test_ratio% 為測試集。
    
    Args:
        df (pd.DataFrame): 完整資料集
        test_ratio (float): 測試集比例，預設 0.2
    
    Returns:
        tuple: (df_train, df_test)
    """
    
    split_point = int(len(df) * (1 - test_ratio))
    df_train = df[:split_point].reset_index(drop=True)
    df_test = df[split_point:].reset_index(drop=True)
    
    print(f"\n時間切分結果:")
    print(f"  訓練集: {len(df_train)} 筆 ({100*(1-test_ratio):.1f}%)")
    print(f"  測試集: {len(df_test)} 筆 ({100*test_ratio:.1f}%)")
    
    if 'datetime' in df.columns:
        print(f"  訓練集時間: {df_train['datetime'].min()} ~ {df_train['datetime'].max()}")
        print(f"  測試集時間: {df_test['datetime'].min()} ~ {df_test['datetime'].max()}")
    
    return df_train, df_test


import numpy as np
