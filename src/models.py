"""
模型建立模組 - 建立和訓練三種回歸模型
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("警告: 未安裝 XGBoost，請執行: pip install xgboost")

from config import RF_PARAMS, XGBOOST_PARAMS, RANDOM_STATE


def build_linear_regression():
    """
    建立線性迴歸模型。
    
    Returns:
        LinearRegression: 線性迴歸模型
    """
    
    print("建立 Linear Regression 模型...")
    model = LinearRegression()
    return model


def build_random_forest():
    """
    建立隨機森林迴歸模型。
    
    參數：
    - n_estimators: 300
    - random_state: 42
    - n_jobs: -1 (使用所有 CPU 核心)
    
    Returns:
        RandomForestRegressor: 隨機森林迴歸模型
    """
    
    print("建立 Random Forest Regressor 模型...")
    model = RandomForestRegressor(**RF_PARAMS)
    return model


def build_xgboost():
    """
    建立 XGBoost 迴歸模型。
    
    參數：
    - n_estimators: 300
    - learning_rate: 0.05
    - max_depth: 5
    - random_state: 42
    - n_jobs: -1
    
    Returns:
        XGBRegressor: XGBoost 迴歸模型 (若已安裝)
        None: 若 XGBoost 未安裝
    """
    
    if not XGBOOST_AVAILABLE:
        print("XGBoost 未安裝，跳過此模型")
        return None
    
    print("建立 XGBoost Regressor 模型...")
    model = XGBRegressor(**XGBOOST_PARAMS, verbosity=0)
    return model


def build_lightgbm():
    """
    建立 LightGBM 迴歸模型。
    
    參數：
    - n_estimators: 300
    - learning_rate: 0.05
    - max_depth: 5
    - random_state: 42
    - verbose: -1
    
    Returns:
        LGBMRegressor: LightGBM 迴歸模型 (若已安裝)
        None: 若 LightGBM 未安裝
    """
    
    try:
        from lightgbm import LGBMRegressor
        print("建立 LightGBM Regressor 模型...")
        model = LGBMRegressor(
            n_estimators=300, 
            learning_rate=0.05, 
            max_depth=5, 
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1
        )
        return model
    except ImportError:
        print("LightGBM 未安裝，跳過此模型")
        return None


def build_ridge():
    """
    建立 Ridge Regression 模型（帶 L2 正則化的線性迴歸）。
    
    Returns:
        Ridge: Ridge 迴歸模型
    """
    from sklearn.linear_model import Ridge
    print("建立 Ridge Regression 模型...")
    model = Ridge(alpha=1.0)
    return model


def build_lasso():
    """
    建立 Lasso Regression 模型（帶 L1 正則化的線性迴歸）。
    
    Returns:
        Lasso: Lasso 迴歸模型
    """
    from sklearn.linear_model import Lasso
    print("建立 Lasso Regression 模型...")
    model = Lasso(alpha=0.01, max_iter=10000)
    return model


def build_svr():
    """
    建立 SVR 模型（支援向量迴歸）。
    
    Returns:
        SVR: Support Vector Regression 模型
    """
    from sklearn.svm import SVR
    print("建立 SVR 模型...")
    model = SVR(kernel='rbf', C=100, epsilon=0.1)
    return model


def train_models(X_train, y_train):
    """
    訓練所有模型 (基礎 + 升級)。
    
    Args:
        X_train (pd.DataFrame): 訓練集特徵
        y_train (pd.Series): 訓練集目標值
    
    Returns:
        dict: 已訓練的模型字典
               {
                'Linear Regression': model,
                'Ridge': model,
                'Lasso': model,
                'Random Forest': model,
                'SVR': model,
                'XGBoost': model,
                'LightGBM': model
               }
    """
    
    print("\n" + "="*60)
    print("開始訓練模型")
    print("="*60)
    
    models = {}
    model_count = 0
    
    # ===== 基礎線性模型 =====
    print("\n[線性迴歸模型]")
    
    # 1. 線性迴歸
    print("\n[1] 訓練 Linear Regression...")
    lr_model = build_linear_regression()
    lr_model.fit(X_train, y_train)
    models["Linear Regression"] = lr_model
    model_count += 1
    print("✓ Linear Regression 訓練完成")
    
    # 2. Ridge Regression
    print("\n[2] 訓練 Ridge Regression...")
    ridge_model = build_ridge()
    ridge_model.fit(X_train, y_train)
    models["Ridge"] = ridge_model
    model_count += 1
    print("✓ Ridge Regression 訓練完成")
    
    # 3. Lasso Regression
    print("\n[3] 訓練 Lasso Regression...")
    lasso_model = build_lasso()
    lasso_model.fit(X_train, y_train)
    models["Lasso"] = lasso_model
    model_count += 1
    print("✓ Lasso Regression 訓練完成")
    
    # ===== 非線性模型 =====
    print("\n[非線性模型]")
    
    # 4. 隨機森林
    print("\n[4] 訓練 Random Forest Regressor...")
    rf_model = build_random_forest()
    rf_model.fit(X_train, y_train)
    models["Random Forest"] = rf_model
    model_count += 1
    print("✓ Random Forest 訓練完成")
    
    # 5. SVR
    print("\n[5] 訓練 SVR...")
    svr_model = build_svr()
    svr_model.fit(X_train, y_train)
    models["SVR"] = svr_model
    model_count += 1
    print("✓ SVR 訓練完成")
    
    # 6. XGBoost
    print("\n[6] 訓練 XGBoost Regressor...")
    xgb_model = build_xgboost()
    if xgb_model is not None:
        xgb_model.fit(X_train, y_train)
        models["XGBoost"] = xgb_model
        model_count += 1
        print("✓ XGBoost 訓練完成")
    else:
        print("✗ XGBoost 訓練跳過（未安裝）")
    
    # 7. LightGBM
    print("\n[7] 訓練 LightGBM Regressor...")
    lgb_model = build_lightgbm()
    if lgb_model is not None:
        lgb_model.fit(X_train, y_train)
        models["LightGBM"] = lgb_model
        model_count += 1
        print("✓ LightGBM 訓練完成")
    else:
        print("✗ LightGBM 訓練跳過（未安裝）")
    
    print("\n" + "="*60)
    print(f"模型訓練完成 (共 {model_count} 個模型)")
    print(f"模型列表: {list(models.keys())}")
    print("="*60)
    
    return models


def predict(models, X_data):
    """
    使用所有模型進行預測。
    
    Args:
        models (dict): 模型字典
        X_data (pd.DataFrame): 特徵資料
    
    Returns:
        dict: 預測結果字典
              {'Linear Regression': y_pred1, 
               'Random Forest': y_pred2, 
               'XGBoost': y_pred3}
    """
    
    predictions = {}
    for model_name, model in models.items():
        predictions[model_name] = model.predict(X_data)
    
    return predictions


def display_model_info(models):
    """
    顯示每個模型的詳細資訊。
    
    Args:
        models (dict): 模型字典
    """
    
    print("\n" + "="*60)
    print("已訓練模型資訊")
    print("="*60)
    
    for i, (model_name, model) in enumerate(models.items(), 1):
        print(f"\n[{i}] {model_name}")
        print(f"  模型類別: {type(model).__name__}")
        
        # 顯示主要參數
        params = model.get_params()
        
        # 篩選重要參數（不全部顯示，以免太冗長）
        important_keys = [
            'alpha', 'C', 'epsilon', 'kernel', 'gamma',
            'n_estimators', 'max_depth', 'learning_rate', 'random_state',
            'n_jobs', 'verbose'
        ]
        
        print(f"  主要參數:")
        for key in important_keys:
            if key in params:
                print(f"    - {key}: {params[key]}")
        
        # 顯示模型大小 (若有)
        if hasattr(model, 'n_features_in_'):
            print(f"  輸入特徵數: {model.n_features_in_}")
    
    print("\n" + "="*60)