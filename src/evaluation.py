"""
模型評估模組 - 計算評估指標並比較模型
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred, model_name="", dataset_name="訓練集"):
    """
    計算單一模型的評估指標 (MAE, RMSE, R2)。
    
    Args:
        y_true (array-like): 真實值
        y_pred (array-like): 預測值
        model_name (str): 模型名稱
        dataset_name (str): 資料集名稱 (訓練集/測試集)
    
    Returns:
        dict: 包含 MAE, RMSE, R2 的字典
    """
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def evaluate_models(models, X_train, y_train, X_test, y_test, target_col):
    """
    評估所有模型的訓練集和測試集表現。
    
    Args:
        models (dict): 模型字典
        X_train (pd.DataFrame): 訓練集特徵
        y_train (pd.Series): 訓練集目標值
        X_test (pd.DataFrame): 測試集特徵
        y_test (pd.Series): 測試集目標值
        target_col (str): 目標欄位名稱
    
    Returns:
        dict: 評估結果
        pd.DataFrame: 評估結果表格
    """
    
    print("\n" + "="*60)
    print("開始模型評估")
    print("="*60)
    
    evaluation_results = []
    predictions_dict = {}
    
    for model_name, model in models.items():
        print(f"\n評估 {model_name}...")
        
        # 訓練集預測
        y_train_pred = model.predict(X_train)
        train_metrics = calculate_metrics(y_train, y_train_pred, model_name, "訓練集")
        
        # 測試集預測
        y_test_pred = model.predict(X_test)
        test_metrics = calculate_metrics(y_test, y_test_pred, model_name, "測試集")
        
        # 儲存預測結果
        predictions_dict[model_name] = {
            "train_pred": y_train_pred,
            "test_pred": y_test_pred
        }
        
        # 組合結果
        result = {
            "model": model_name,
            "target": target_col,
            "train_MAE": train_metrics["MAE"],
            "train_RMSE": train_metrics["RMSE"],
            "train_R2": train_metrics["R2"],
            "test_MAE": test_metrics["MAE"],
            "test_RMSE": test_metrics["RMSE"],
            "test_R2": test_metrics["R2"]
        }
        
        evaluation_results.append(result)
        
        # 顯示結果
        print(f"  訓練集 - MAE: {train_metrics['MAE']:.4f}, "
              f"RMSE: {train_metrics['RMSE']:.4f}, "
              f"R2: {train_metrics['R2']:.4f}")
        print(f"  測試集 - MAE: {test_metrics['MAE']:.4f}, "
              f"RMSE: {test_metrics['RMSE']:.4f}, "
              f"R2: {test_metrics['R2']:.4f}")
    
    # 轉為 DataFrame
    evaluation_df = pd.DataFrame(evaluation_results)
    
    print("\n" + "="*60)
    print("評估結果摘要")
    print("="*60)
    print(evaluation_df.to_string(index=False))
    
    return evaluation_results, evaluation_df, predictions_dict


def find_best_model(evaluation_df, metric="test_RMSE"):
    """
    根據指定指標找出最佳模型。
    
    預設使用 test_RMSE (測試集根均方誤差)。
    RMSE 越低越好。
    
    Args:
        evaluation_df (pd.DataFrame): 評估結果表格
        metric (str): 用於判斷最佳模型的指標，預設為 "test_RMSE"
    
    Returns:
        str: 最佳模型名稱
        dict: 最佳模型的評估結果
    """
    
    best_idx = evaluation_df[metric].idxmin()
    best_model_name = evaluation_df.loc[best_idx, "model"]
    best_result = evaluation_df.loc[best_idx].to_dict()
    
    print("\n" + "="*60)
    print("最佳模型選擇")
    print("="*60)
    print(f"根據 {metric} 最低原則")
    print(f"最佳模型: {best_model_name}")
    print(f"測試集 RMSE: {best_result['test_RMSE']:.4f}")
    print(f"測試集 R2: {best_result['test_R2']:.4f}")
    print("="*60)
    
    return best_model_name, best_result


def display_comparison(evaluation_df):
    """
    顯示模型比較表。
    
    Args:
        evaluation_df (pd.DataFrame): 評估結果表格
    """
    
    print("\n" + "="*60)
    print("詳細模型比較")
    print("="*60)
    print(evaluation_df.to_string(index=False))
