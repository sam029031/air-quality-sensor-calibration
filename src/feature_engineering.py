"""
特徵工程模組 - 建立時間相關特徵和其他衍生特徵
"""

import pandas as pd
import numpy as np


def parse_datetime(df, date_col="Date", time_col="Time"):
    """
    將 Date 和 Time 欄位合併成 datetime。
    
    Args:
        df (pd.DataFrame): 輸入資料框
        date_col (str): 日期欄位名稱
        time_col (str): 時間欄位名稱
    
    Returns:
        pd.DataFrame: 包含 datetime 的資料框
    """
    
    print("\n合併 Date 和 Time 欄位成 datetime...")
    
    # 檢查欄位是否存在
    if date_col not in df.columns or time_col not in df.columns:
        print(f"警告: 找不到 '{date_col}' 或 '{time_col}' 欄位")
        print(f"現有欄位: {list(df.columns)}")
        return df
    
    try:
        # 合併 Date 和 Time
        # 時間格式: HH.MM.SS（如 18.00.00）
        time_clean = df[time_col].astype(str).str.replace('.', '', regex=False)
        
        df['datetime'] = pd.to_datetime(
            df[date_col].astype(str) + " " + time_clean,
            format="%d/%m/%Y %H%M%S",
            errors="coerce"
        )
        
        # 移除無法轉換的行
        valid_count = df['datetime'].notna().sum()
        invalid_count = df['datetime'].isna().sum()
        print(f"成功轉換: {valid_count} 筆")
        if invalid_count > 0:
            print(f"轉換失敗: {invalid_count} 筆")
            df = df.dropna(subset=['datetime'])
        
        # 按時間排序
        df = df.sort_values('datetime').reset_index(drop=True)
        print(f"資料時間範圍: {df['datetime'].min()} 到 {df['datetime'].max()}")
        
    except Exception as e:
        print(f"無法合併 datetime: {e}")
    
    return df


def extract_time_features(df, datetime_col="datetime"):
    """
    從 datetime 提取時間特徵。
    
    提取的特徵：
    - hour: 小時 (0-23)
    - day_of_week: 星期幾 (0=星期一, 6=星期日)
    - month: 月份 (1-12)
    - is_weekend: 是否週末 (0 或 1)
    - season: 季節 (1=春, 2=夏, 3=秋, 4=冬)
    - rush_hour: 尖峰時段 (1 if 7-9 or 17-19, else 0)
    
    Args:
        df (pd.DataFrame): 包含 datetime 欄位的資料框
        datetime_col (str): datetime 欄位名稱
    
    Returns:
        pd.DataFrame: 包含時間特徵的資料框
    """
    
    print("\n提取時間特徵...")
    
    if datetime_col not in df.columns:
        print(f"警告: 找不到 '{datetime_col}' 欄位")
        return df
    
    # 小時
    df['hour'] = df[datetime_col].dt.hour
    
    # 星期幾 (0=星期一, 6=星期日)
    df['day_of_week'] = df[datetime_col].dt.dayofweek
    
    # 月份
    df['month'] = df[datetime_col].dt.month
    
    # 是否週末 (0=週一-五, 1=六日)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # 季節 (1=春3-5, 2=夏6-8, 3=秋9-11, 4=冬12,1,2)
    def get_season(month):
        if month in [3, 4, 5]:
            return 1  # 春
        elif month in [6, 7, 8]:
            return 2  # 夏
        elif month in [9, 10, 11]:
            return 3  # 秋
        else:
            return 4  # 冬
    
    df['season'] = df['month'].apply(get_season)
    
    # 尖峰時段 (早上7-9點或傍晚17-19點)
    df['rush_hour'] = df['hour'].apply(lambda h: 1 if (7 <= h <= 9) or (17 <= h <= 19) else 0)
    
    print("時間特徵提取完成")
    print(f"  - hour: {df['hour'].min()}-{df['hour'].max()}")
    print(f"  - day_of_week: {df['day_of_week'].unique()}")
    print(f"  - month: {sorted(df['month'].unique())}")
    print(f"  - is_weekend: {df['is_weekend'].unique()}")
    print(f"  - season: {sorted(df['season'].unique())}")
    print(f"  - rush_hour: {df['rush_hour'].unique()}")
    
    return df


def feature_engineering(df, date_col="Date", time_col="Time"):
    """
    執行完整的特徵工程流程。
    
    包含：
    1. 解析 datetime
    2. 提取時間特徵
    
    Args:
        df (pd.DataFrame): 輸入資料框
        date_col (str): 日期欄位名稱
        time_col (str): 時間欄位名稱
    
    Returns:
        pd.DataFrame: 包含特徵工程後資料的資料框
    """
    
    print("\n" + "="*60)
    print("開始特徵工程")
    print("="*60)
    
    # 1. 解析 datetime
    df = parse_datetime(df, date_col=date_col, time_col=time_col)
    
    # 2. 提取時間特徵
    df = extract_time_features(df, datetime_col="datetime")
    
    print("\n特徵工程完成")
    print(f"新增特徵: hour, day_of_week, month, is_weekend, season, rush_hour")
    
    return df


def create_lag_features(df, target_col="CO(GT)", lags=[1, 3, 24], dropna=True):
    """
    為目標欄位創建滯後特徵（lag features）。
    
    利用前一小時、前3小時、前24小時的目標污染物濃度作為預測特徵。
    這對時序預測非常有效，因為污染物濃度有強烈的時間自相關性。
    
    Args:
        df (pd.DataFrame): 輸入資料框（必須按時間排序）
        target_col (str): 目標欄位名稱
        lags (list): 滯後小時數列表，預設 [1, 3, 24]
        dropna (bool): 是否移除含 NaN 的行，預設 True
    
    Returns:
        pd.DataFrame: 包含滯後特徵的資料框
    """
    
    print(f"\n建立滯後特徵 (Lag Features)...")
    
    if target_col not in df.columns:
        print(f"警告: 找不到 '{target_col}' 欄位")
        return df
    
    # 確保資料按時間排序
    if 'datetime' in df.columns:
        df = df.sort_values('datetime').reset_index(drop=True)
    
    initial_len = len(df)
    
    # 為每個滯後時間建立特徵
    for lag in lags:
        col_name = f"{target_col}_lag_{lag}h"
        df[col_name] = df[target_col].shift(lag)
        print(f"  ✓ 建立特徵: {col_name}")
    
    # 移除 NaN（如果設定 dropna=True）
    if dropna:
        df = df.dropna()
        removed = initial_len - len(df)
        print(f"\n滯後特徵建立完成")
        print(f"  - 新增特徵: {[f'{target_col}_lag_{lag}h' for lag in lags]}")
        print(f"  - 移除 NaN 列數: {removed}")
        print(f"  - 保留資料筆數: {len(df)}")
    else:
        nan_count = df[[f'{target_col}_lag_{lag}h' for lag in lags]].isna().sum().sum()
        print(f"\n滯後特徵建立完成")
        print(f"  - 新增特徵: {[f'{target_col}_lag_{lag}h' for lag in lags]}")
        print(f"  - 新增 NaN 個數: {nan_count}")
        print(f"  - 資料筆數: {len(df)}")
    
    return df