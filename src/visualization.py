"""
視覺化模組 - 繪製各種圖表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.font_manager as fm

# 設定中文字體（備援列表，會依系統可用字型自動回退）
preferred_fonts = [
    'Microsoft JhengHei',  # Windows 繁體
    'Microsoft YaHei',     # Windows 簡體
    'SimHei',              # 常見中文字型（部分環境）
    'Noto Sans CJK TC',    # Noto CJK 繁體
    'Noto Sans CJK JP',    # Noto 日文（CJK 援助）
    'Arial Unicode MS',
    'DejaVu Sans'
]

available_fonts = {f.name for f in fm.fontManager.ttflist}
font_list = [f for f in preferred_fonts if f in available_fonts]
if font_list:
    plt.rcParams['font.sans-serif'] = font_list
else:
    # 最後保險回退到 matplotlib 的預設可用字型
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

from config import FIGURES_DIR


def plot_time_series_comparison(df_test, y_test, y_pred, target_col, best_model_name):
    """
    繪製時序圖：真實值 vs 校正預測值
    
    Args:
        df_test (pd.DataFrame): 測試集資料
        y_test (pd.Series): 真實值
        y_pred (array): 預測值
        target_col (str): 目標欄位名稱
        best_model_name (str): 最佳模型名稱
    """
    
    print(f"\n繪製時序圖...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = range(len(y_test))
    ax.plot(x, y_test.values, label="真實值", marker="o", markersize=3, linewidth=1, alpha=0.7)
    ax.plot(x, y_pred, label="校正預測值", marker="s", markersize=3, linewidth=1, alpha=0.7)
    
    ax.set_xlabel("時間序列")
    ax.set_ylabel(f"{target_col} 濃度")
    ax.set_title(f"{target_col} 真實值 vs 校正預測值（{best_model_name}）")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    filename = FIGURES_DIR / f"time_series_comparison_{target_col}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ 已儲存: {filename}")


def plot_scatter_comparison(y_test, y_pred, target_col, best_model_name):
    """
    繪製散佈圖：真實值 vs 校正預測值
    
    Args:
        y_test (pd.Series): 真實值
        y_pred (array): 預測值
        target_col (str): 目標欄位名稱
        best_model_name (str): 最佳模型名稱
    """
    
    print(f"繪製散佈圖...")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(y_test, y_pred, alpha=0.6, s=20)
    
    # 繪製完美預測線
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="完美預測")
    
    ax.set_xlabel(f"真實 {target_col}")
    ax.set_ylabel(f"預測 {target_col}")
    ax.set_title(f"{target_col} 真實值 vs 預測值散佈圖（{best_model_name}）")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    filename = FIGURES_DIR / f"scatter_comparison_{target_col}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ 已儲存: {filename}")


def plot_residuals(y_test, y_pred, target_col, best_model_name):
    """
    繪製殘差圖
    
    Args:
        y_test (pd.Series): 真實值
        y_pred (array): 預測值
        target_col (str): 目標欄位名稱
        best_model_name (str): 最佳模型名稱
    """
    
    print(f"繪製殘差圖...")
    
    residuals = y_test.values - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 殘差 vs 預測值
    axes[0].scatter(y_pred, residuals, alpha=0.6, s=20)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel(f"預測 {target_col}")
    axes[0].set_ylabel("殘差")
    axes[0].set_title(f"殘差 vs 預測值")
    axes[0].grid(True, alpha=0.3)
    
    # 殘差分佈
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel("殘差")
    axes[1].set_ylabel("頻率")
    axes[1].set_title(f"殘差分佈")
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].grid(True, alpha=0.3)
    
    filename = FIGURES_DIR / f"residuals_{target_col}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ 已儲存: {filename}")


def plot_model_comparison_bar(evaluation_df, target_col):
    """
    繪製模型比較長條圖 (test RMSE)
    
    Args:
        evaluation_df (pd.DataFrame): 評估結果表格
        target_col (str): 目標欄位名稱
    """
    
    print(f"繪製模型 test RMSE 比較圖...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = evaluation_df["model"]
    test_rmse = evaluation_df["test_RMSE"]
    
    colors = ['green' if x == test_rmse.min() else 'steelblue' for x in test_rmse]
    bars = ax.bar(models, test_rmse, color=colors, edgecolor='black', alpha=0.7)
    
    ax.set_ylabel("Test RMSE")
    ax.set_xlabel("模型")
    ax.set_title(f"{target_col} 模型 Test RMSE 比較")
    ax.grid(True, alpha=0.3, axis='y')
    
    # 在每個長條上顯示數值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    filename = FIGURES_DIR / f"model_comparison_rmse_{target_col}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ 已儲存: {filename}")


def plot_feature_importance(model, feature_cols, target_col, best_model_name):
    """
    繪製特徵重要性圖 (適用於 Random Forest 或 XGBoost)
    
    Args:
        model: 已訓練的模型（Random Forest 或 XGBoost）
        feature_cols (list): 特徵名稱列表
        target_col (str): 目標欄位名稱
        best_model_name (str): 最佳模型名稱
    """
    
    print(f"繪製特徵重要性圖...")
    
    # 檢查模型是否有 feature_importances_ 屬性
    if not hasattr(model, 'feature_importances_'):
        print("此模型不支援特徵重要性")
        return
    
    importances = model.feature_importances_
    
    # 排序
    indices = np.argsort(importances)[::-1][:15]  # 取前 15 個
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_features = [feature_cols[i] for i in indices]
    top_importances = importances[indices]
    
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_importances, edgecolor='black', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.set_xlabel("重要性")
    ax.set_title(f"{target_col} 特徵重要性（{best_model_name}）")
    ax.grid(True, alpha=0.3, axis='x')
    
    filename = FIGURES_DIR / f"feature_importance_{target_col}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ 已儲存: {filename}")


def plot_all_results(models, X_train, y_train, X_test, y_test, 
                     evaluation_df, target_col, best_model_name, df_test=None):
    """
    繪製所有視覺化圖表
    
    Args:
        models (dict): 模型字典
        X_train (pd.DataFrame): 訓練集特徵
        y_train (pd.Series): 訓練集目標值
        X_test (pd.DataFrame): 測試集特徵
        y_test (pd.Series): 測試集目標值
        evaluation_df (pd.DataFrame): 評估結果表格
        target_col (str): 目標欄位名稱
        best_model_name (str): 最佳模型名稱
        df_test (pd.DataFrame): 測試集完整資料 (用於時間軸和校正圖)
    """
    
    print("\n" + "="*60)
    print("開始繪製視覺化圖表")
    print("="*60)
    
    # 取得最佳模型
    best_model = models[best_model_name]
    
    # 預測值
    y_pred = best_model.predict(X_test)
    
    # ===== 基礎圖表 =====
    # 時序圖
    plot_time_series_comparison(X_test, y_test, y_pred, target_col, best_model_name)
    
    # 散佈圖
    plot_scatter_comparison(y_test, y_pred, target_col, best_model_name)
    
    # 殘差圖
    plot_residuals(y_test, y_pred, target_col, best_model_name)
    
    # 模型比較圖
    plot_model_comparison_bar(evaluation_df, target_col)
    
    # 特徵重要性 (若支援)
    if best_model_name in ["Random Forest", "XGBoost", "LightGBM"]:
        plot_feature_importance(best_model, X_test.columns.tolist(), target_col, best_model_name)
    
    # ===== EDA & 季節性分析 =====
    print("\n繪製 EDA 分析圖...")
    
    # 相關係數熱力圖
    try:
        plot_correlation_heatmap(X_test, target_col)
    except Exception as e:
        print(f"⚠️ 相關係數熱力圖繪製失敗: {e}")
    
    # 月份箱型圖 (需要有 datetime 欄位)
    if df_test is not None and 'datetime' in df_test.columns:
        try:
            plot_monthly_boxplot(df_test, target_col, datetime_col='datetime')
        except Exception as e:
            print(f"⚠️ 月份箱型圖繪製失敗: {e}")
        
        # 小時 × 星期幾熱力圖
        try:
            plot_hourly_by_dow_heatmap(df_test, target_col, datetime_col='datetime')
        except Exception as e:
            print(f"⚠️ 熱力圖繪製失敗: {e}")
        
        # 殘差 vs 時間
        try:
            plot_residuals_vs_time(df_test, y_test, y_pred, target_col, datetime_col='datetime')
        except Exception as e:
            print(f"⚠️ 殘差 vs 時間圖繪製失敗: {e}")
    
    # ===== 模型解釋性圖 =====
    print("\n繪製模型解釋性圖...")
    
    # Learning Curve (訓練較長時間但很有用)
    if best_model_name in ["XGBoost", "LightGBM", "Random Forest"]:
        try:
            plot_learning_curve(best_model, X_train, y_train, target_col, best_model_name)
        except Exception as e:
            print(f"⚠️ Learning Curve 繪製失敗: {e}")
    
    # SHAP 圖 (樹基模型特適用)
    if best_model_name in ["XGBoost", "LightGBM", "Random Forest"]:
        try:
            plot_shap_summary(best_model, X_test, target_col, best_model_name)
        except Exception as e:
            print(f"⚠️ SHAP 圖繪製失敗 (可能需要安裝 shap): {e}")
    
    # ===== 校正效果對比 =====
    print("\n繪製校正效果對比圖...")
    if df_test is not None:
        try:
            # 需要從 config 引入 SENSOR_TO_MODEL
            from config import SENSOR_TO_MODEL
            plot_calibration_before_after(df_test, y_test, y_pred, target_col, 
                                         target_col, SENSOR_TO_MODEL)
        except Exception as e:
            print(f"⚠️ 校正對比圖繪製失敗: {e}")
    
    print("\n" + "="*60)
    print("所有圖表繪製完成")
    print("="*60)


def plot_correlation_heatmap(X_test, target_col):
    """
    特徵相關係數熱力圖，顯示為什麼某些感測器重要
    及感測器間的交叉敏感性
    """
    print(f"繪製相關係數熱力圖...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_matrix = X_test.corr()
    sns.heatmap(corr_matrix, annot=False, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax, square=True, cbar_kws={'label': '相關係數'})
    ax.set_title(f'特徵相關係數矩陣')
    
    filename = FIGURES_DIR / f"correlation_heatmap_{target_col}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已儲存: {filename}")


def plot_monthly_boxplot(df, target_col, datetime_col='datetime'):
    """
    各月份污染物濃度箱型圖，呈現季節性變化
    冬季逆溫層效應會很明顯
    """
    print(f"繪製月份箱型圖...")
    
    df_temp = df.copy()
    df_temp['month'] = df_temp[datetime_col].dt.month
    
    fig, ax = plt.subplots(figsize=(12, 6))
    df_temp.boxplot(column=target_col, by='month', ax=ax)
    ax.set_title(f'{target_col} 月份箱型圖 (季節性分析)')
    ax.set_xlabel('月份')
    ax.set_ylabel(target_col)
    plt.suptitle('')  # 移除自動標題
    
    filename = FIGURES_DIR / f"monthly_boxplot_{target_col}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已儲存: {filename}")


def plot_hourly_by_dow_heatmap(df, target_col, datetime_col='datetime'):
    """
    每小時 × 星期幾的平均污染物濃度熱力圖
    直接視覺化通勤尖峰效應
    """
    print(f"繪製小時×星期幾熱力圖...")
    
    df_temp = df.copy()
    df_temp['hour'] = df_temp[datetime_col].dt.hour
    df_temp['day_of_week'] = df_temp[datetime_col].dt.dayofweek
    
    pivot = df_temp.pivot_table(values=target_col, index='day_of_week', 
                               columns='hour', aggfunc='mean')
    
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(pivot, cmap='YlOrRd', annot=False, ax=ax, cbar_kws={'label': target_col})
    ax.set_title(f'{target_col} 熱力圖：星期幾 × 小時（通勤尖峰效應）')
    ax.set_xlabel('小時')
    ax.set_ylabel('星期幾 (0=週一, 6=週日)')
    
    filename = FIGURES_DIR / f"heatmap_hourly_dow_{target_col}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已儲存: {filename}")


def plot_residuals_vs_time(df_test, y_test, y_pred, target_col, datetime_col='datetime'):
    """
    殘差 vs 時間圖，看是否有週期性
    若有代表模型還沒捕捉到的時序規律
    """
    print(f"繪製殘差 vs 時間圖...")
    
    residuals = y_test.values - y_pred
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(df_test[datetime_col], residuals, alpha=0.6, s=20)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('時間')
    ax.set_ylabel('殘差')
    ax.set_title(f'殘差 vs 時間序列 - {target_col}（檢查未捕捉的週期性）')
    ax.grid(True, alpha=0.3)
    
    filename = FIGURES_DIR / f"residuals_vs_time_{target_col}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已儲存: {filename}")


def plot_learning_curve(model, X_train, y_train, target_col, best_model_name):
    """
    繪製學習曲線，看訓練資料量 vs RMSE
    可直觀說明過擬合的原因
    """
    print(f"繪製 Learning Curve...")
    
    try:
        from sklearn.model_selection import learning_curve
        from sklearn.metrics import mean_squared_error
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=5, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        train_rmse = np.sqrt(-train_scores.mean(axis=1))
        val_rmse = np.sqrt(-val_scores.mean(axis=1))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_sizes, train_rmse, label='訓練 RMSE', marker='o', markersize=5)
        ax.plot(train_sizes, val_rmse, label='驗證 RMSE', marker='s', markersize=5)
        ax.set_xlabel('訓練資料數量')
        ax.set_ylabel('RMSE')
        ax.set_title(f'Learning Curve - {best_model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        filename = FIGURES_DIR / f"learning_curve_{target_col}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 已儲存: {filename}")
    except Exception as e:
        print(f"⚠️ Learning Curve 計算失敗: {e}")


def plot_shap_summary(model, X_test, target_col, best_model_name):
    """
    SHAP summary plot - 顯示每個特徵對預測的貢獻方向（正/負）
    目前最強的模型可解釋性工具
    """
    print(f"繪製 SHAP 特徵貢獻圖...")
    
    try:
        import shap
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        
        filename = FIGURES_DIR / f"shap_summary_{target_col}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 已儲存 SHAP 圖: {filename}")
    except ImportError:
        print("⚠️ 需要安裝 shap: pip install shap")
    except Exception as e:
        print(f"⚠️ SHAP 圖繪製失敗: {e}")


def plot_calibration_before_after(df_test, y_test, y_pred, target_col, 
                                   sensor_col, sensor_to_model):
    """
    校正前後 MAE 對比長條圖
    直接比較原始感測器 MAE vs 校正後 MAE
    視覺衝擊很強，適合放簡報封面頁或結論頁
    """
    print(f"繪製校正前後對比圖...")
    
    try:
        # 原始感測器誤差
        raw_mae = None
        if target_col in sensor_to_model and sensor_to_model[target_col] in df_test.columns:
            raw_mae = np.abs(df_test[sensor_to_model[target_col]].values - y_test.values).mean()
        
        # 校正後誤差
        calibrated_mae = np.abs(y_pred - y_test.values).mean()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if raw_mae is not None:
            bars = ax.bar(['校正前 (Raw)', '校正後 (Calibrated)'], 
                          [raw_mae, calibrated_mae],
                          color=['#d387ab', '#3c6e71'], edgecolor='#353535', linewidth=1.5, alpha=0.85)
        else:
            bars = ax.bar(['校正後 (Calibrated)'], 
                          [calibrated_mae],
                          color=['#3c6e71'], edgecolor='#353535', linewidth=1.5, alpha=0.85)
        
        ax.set_ylabel('MAE (平均絕對誤差)', fontsize=11)
        ax.set_title(f'{target_col} 校正效果對比', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 在長條上顯示數值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        filename = FIGURES_DIR / f"calibration_before_after_{target_col}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 已儲存: {filename}")
    except Exception as e:
        print(f"⚠️ 校正對比圖繪製失敗: {e}")