"""
資料載入模組 - 負責讀取 CSV 或 Excel 格式的空氣品質資料
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from config import DATA_FILE, DATA_FILE_XLSX


def load_data(file_path=None):
    """
    載入空氣品質資料集。
    
    支援 CSV 和 Excel 格式。處理以下特殊格式：
    - 分號 (;) 作為分隔符
    - 逗號 (,) 作為小數點符號
    - 缺失值代碼 -200
    - 自動刪除完全空白的 Unnamed 欄位
    
    Args:
        file_path (str or Path, optional): 資料檔案路徑。
                                           若為 None，則自動尋找預設位置。
    
    Returns:
        pd.DataFrame: 載入的資料集
    
    Raises:
        FileNotFoundError: 若資料檔不存在
    """
    
    if file_path is None:
        # 自動尋找資料檔
        if DATA_FILE.exists():
            file_path = DATA_FILE
        elif DATA_FILE_XLSX.exists():
            file_path = DATA_FILE_XLSX
        else:
            raise FileNotFoundError(
                f"找不到資料檔。\n"
                f"請確保以下檔案其一存在：\n"
                f"  - {DATA_FILE}\n"
                f"  - {DATA_FILE_XLSX}"
            )
    
    file_path = Path(file_path)
    
    print(f"正在載入資料: {file_path}")
    
    try:
        if file_path.suffix.lower() == ".csv":
            # 嘗試用分號作為分隔符（UCI Air Quality 資料集格式）
            df = pd.read_csv(file_path, sep=";", decimal=",")
        elif file_path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"不支援的檔案格式: {file_path.suffix}")
        
        # 刪除完全空白的 Unnamed 欄位
        unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
        if unnamed_cols:
            print(f"移除空白欄位: {unnamed_cols}")
            df = df.drop(columns=unnamed_cols)
        
        print(f"成功載入資料，形狀: {df.shape}")
        print(f"欄位名稱: {list(df.columns)}")
        
        return df
    
    except Exception as e:
        raise IOError(f"載入資料檔時發生錯誤: {e}")


def convert_decimal_comma_to_dot(df):
    """
    將使用逗號作為小數點的欄位轉換為標準小數點格式。
    
    Args:
        df (pd.DataFrame): 輸入資料框
    
    Returns:
        pd.DataFrame: 轉換後的資料框
    """
    
    print("正在轉換小數點格式...")
    
    for col in df.columns:
        if df[col].dtype == object:  # 字符串類型
            try:
                # 嘗試將逗號轉換為點，然後轉換為浮點數
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "."),
                    errors="coerce"
                )
            except:
                pass
    
    return df


def display_data_info(df):
    """
    顯示資料集的基本資訊。
    
    Args:
        df (pd.DataFrame): 資料框
    """
    
    print("\n" + "="*60)
    print("資料集基本資訊")
    print("="*60)
    print(f"資料形狀: {df.shape}")
    print(f"\n欄位數據類型:")
    print(df.dtypes)
    print(f"\n前 5 筆資料:")
    print(df.head())
    print("\n")
