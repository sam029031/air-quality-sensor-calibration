"""
資料前處理模組 - 處理缺失值、異常值和資料清理
"""

import pandas as pd
import numpy as np
from config import MISSING_VALUE_CODE, TARGET_COL, SENSOR_FEATURES, WEATHER_FEATURES


def handle_missing_values(df, missing_code=MISSING_VALUE_CODE):
    """
    將缺失值代碼 -200 轉換為 NaN，並統計缺失情況。
    
    Args:
        df (pd.DataFrame): 輸入資料框
        missing_code (int): 缺失值代碼，預設為 -200
    
    Returns:
        pd.DataFrame: 處理後的資料框
    """
    
    print(f"\n正在處理缺失值（代碼: {missing_code})...")
    
    # 將缺失值代碼轉換為 NaN
    df = df.replace(missing_code, np.nan)
    
    # 顯示缺失值統計
    display_missing_statistics(df)
    
    return df


def display_missing_statistics(df):
    """
    統計並顯示缺失值情況，並儲存成 CSV。
    
    Args:
        df (pd.DataFrame): 資料框
    """
    
    print("\n" + "="*60)
    print("缺失值統計")
    print("="*60)
    
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    
    missing_df = pd.DataFrame({
        "欄位": df.columns,
        "缺失數量": missing_count.values,
        "缺失比例(%)": missing_percent.values
    })
    
    # 只顯示有缺失值的欄位
    missing_df = missing_df[missing_df["缺失數量"] > 0].sort_values("缺失比例(%)", ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df.to_string(index=False))
    else:
        print("無缺失值")
    
    return missing_df


def remove_target_missing(df, target_col):
    """
    刪除目標欄位為缺失值的資料列。
    
    Args:
        df (pd.DataFrame): 輸入資料框
        target_col (str): 目標欄位名稱
    
    Returns:
        pd.DataFrame: 處理後的資料框
    """
    
    print(f"\n刪除目標欄位 '{target_col}' 為缺失值的資料列...")
    
    initial_rows = len(df)
    df = df.dropna(subset=[target_col])
    removed_rows = initial_rows - len(df)
    
    print(f"刪除 {removed_rows} 筆資料")
    print(f"剩餘 {len(df)} 筆資料")
    
    return df


def impute_features_median(df, feature_cols):
    """
    使用中位數補填特徵欄位的缺失值。
    
    Args:
        df (pd.DataFrame): 輸入資料框
        feature_cols (list): 特徵欄位名稱列表
    
    Returns:
        pd.DataFrame: 處理後的資料框
    """
    
    print("\n使用中位數補填特徵欄位的缺失值...")
    
    for col in feature_cols:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                print(f"  - {col}: 補填 {missing_count} 個缺失值，使用中位數 {median_value:.4f}")
    
    return df


def preprocess_data(df, target_col=TARGET_COL):
    """
    執行完整的資料前處理流程。
    
    包含：
    1. 處理缺失值代碼
    2. 刪除目標欄位為缺失值的資料
    3. 用中位數補填特徵欄位缺失值
    4. 移除包含 NaN 的行
    
    Args:
        df (pd.DataFrame): 輸入資料框
        target_col (str): 目標欄位名稱
    
    Returns:
        pd.DataFrame: 完整前處理後的資料框
    """
    
    print("\n" + "="*60)
    print("開始資料前處理")
    print("="*60)
    
    # 1. 處理缺失值代碼
    df = handle_missing_values(df)
    
    # 1.5. 移除 NMHC(GT) 列（缺失值過高 88%）
    if "NMHC(GT)" in df.columns:
        print("\n移除 NMHC(GT) 欄位（缺失值過高 88%）")
        df = df.drop(columns=["NMHC(GT)"])
    
    # 2. 刪除目標欄位為缺失值的資料
    df = remove_target_missing(df, target_col)
    
    # 3. 補填特徵欄位缺失值
    all_features = SENSOR_FEATURES + WEATHER_FEATURES
    df = impute_features_median(df, all_features)
    
    # 4. 移除 datetime 為 NaN 的行（無法解析時間的行）
    initial_rows = len(df)
    df = df.dropna(subset=['datetime'])
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        print(f"\n移除無效時間記錄的資料列: {removed_rows}")
    
    # 補填時間特徵的 NaN（確保所有時間特徵都有效）
    time_features_to_fill = ['hour', 'day_of_week', 'month', 'is_weekend', 'season', 'rush_hour']
    for col in time_features_to_fill:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col].fillna(0, inplace=True)
    
    print("\n資料前處理完成")
    print(f"最終資料形狀: {df.shape}")
    print(f"缺失值總數: {df.isnull().sum().sum()}")
    
    return df
