"""
Visualization script for XGBoost body measurement prediction model.
Creates plots to visualize model performance and insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from train_predictor_model import BodyMeasurementPredictor
import os
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import MODEL_FILES_DIR, FIGURES_DIR

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_prediction_vs_actual():
    """
    Create scatter plots comparing predictions vs actual values for each target.
    """
    # Load model
    predictor = BodyMeasurementPredictor()
    predictor.define_measurements()
    
    # Load data
    dataset_path = r'C:\Users\LENOVO\Desktop\Kekious_Maximus\human-body-reshape-DL-paper\data\datasets\CAESAR_full_imputation_dataset.csv'
    df = predictor.load_data(dataset_path)
    
    # Prepare data
    X, y = predictor.prepare_data(df)
    
    # Load trained model
    import glob
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    model_dir = os.path.join(project_root, 'data', 'model_files')
    model_files = glob.glob(os.path.join(model_dir, 'xgboost_body_predictor_*.pkl'))
    latest_model = max(model_files, key=os.path.getmtime)
    
    predictor.load_model(latest_model)
    
    # Make predictions
    y_pred = predictor.predict(X)
    
    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, target in enumerate(predictor.target_measurements):
        ax = axes[i]
        
        # Get actual and predicted values
        actual = y.iloc[:, i].values
        predicted = y_pred[:, i]
        
        # Scatter plot
        ax.scatter(actual, predicted, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(actual, predicted)
        
        ax.set_xlabel('Actual Value', fontsize=10)
        ax.set_ylabel('Predicted Value', fontsize=10)
        ax.set_title(f'{target}\nR² = {r2:.3f}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, 'xgboost_predictions_vs_actual.png'), dpi=300, bbox_inches='tight')
    print(f"\nSaved: {os.path.join(FIGURES_DIR, 'xgboost_predictions_vs_actual.png')}")
    plt.close()


def plot_feature_importance():
    """
    Create bar plots showing feature importance for each target measurement.
    """
    # Load model
    predictor = BodyMeasurementPredictor()
    predictor.define_measurements()
    
    import glob
    model_files = glob.glob(os.path.join(MODEL_FILES_DIR, 'xgboost_body_predictor_*.pkl'))
    latest_model = max(model_files, key=os.path.getmtime)
    
    predictor.load_model(latest_model)
    
    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, target in enumerate(predictor.target_measurements):
        ax = axes[i]
        
        # Get feature importances
        estimator = predictor.model.estimators_[i]
        importances = estimator.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': predictor.proxy_measurements,
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        # Bar plot
        colors = plt.cm.viridis(importance_df['Importance'] / importance_df['Importance'].max())
        ax.barh(range(len(importance_df)), importance_df['Importance'], color=colors)
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['Feature'], fontsize=8)
        ax.set_xlabel('Importance', fontsize=10)
        ax.set_title(target, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, 'xgboost_feature_importance.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(FIGURES_DIR, 'xgboost_feature_importance.png')}")
    plt.close()


def plot_error_distribution():
    """
    Plot the distribution of prediction errors for each target measurement.
    """
    # Load model
    predictor = BodyMeasurementPredictor()
    predictor.define_measurements()
    
    # Load data
    dataset_path = r'C:\Users\LENOVO\Desktop\Kekious_Maximus\human-body-reshape-DL-paper\data\datasets\CAESAR_full_imputation_dataset.csv'
    df = predictor.load_data(dataset_path)
    
    # Prepare data
    X, y = predictor.prepare_data(df)
    
    # Load trained model
    import glob
    model_files = glob.glob(os.path.join(MODEL_FILES_DIR, 'xgboost_body_predictor_*.pkl'))
    latest_model = max(model_files, key=os.path.getmtime)
    
    predictor.load_model(latest_model)
    
    # Make predictions
    y_pred = predictor.predict(X)
    
    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, target in enumerate(predictor.target_measurements):
        ax = axes[i]
        
        # Calculate errors
        errors = y.iloc[:, i].values - y_pred[:, i]
        
        # Histogram
        ax.hist(errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.axvline(errors.mean(), color='green', linestyle='--', linewidth=2, 
                  label=f'Mean Error: {errors.mean():.2f}')
        
        ax.set_xlabel('Prediction Error', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{target}\nStd: {errors.std():.2f}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, 'xgboost_error_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(FIGURES_DIR, 'xgboost_error_distribution.png')}")
    plt.close()


def plot_performance_summary():
    """
    Create a summary bar chart of model performance metrics.
    """
    # Load model
    predictor = BodyMeasurementPredictor()
    predictor.define_measurements()
    
    # Load data
    dataset_path = r'C:\Users\LENOVO\Desktop\Kekious_Maximus\human-body-reshape-DL-paper\data\datasets\CAESAR_full_imputation_dataset.csv'
    df = predictor.load_data(dataset_path)
    
    # Prepare data
    X, y = predictor.prepare_data(df)
    
    # Load trained model
    import glob
    model_files = glob.glob(os.path.join(MODEL_FILES_DIR, 'xgboost_body_predictor_*.pkl'))
    latest_model = max(model_files, key=os.path.getmtime)
    
    predictor.load_model(latest_model)
    
    # Make predictions
    y_pred = predictor.predict(X)
    
    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, r2_score
    
    metrics_data = []
    for i, target in enumerate(predictor.target_measurements):
        mae = mean_absolute_error(y.iloc[:, i], y_pred[:, i])
        r2 = r2_score(y.iloc[:, i], y_pred[:, i])
        metrics_data.append({'Target': target, 'MAE': mae, 'R²': r2})
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # MAE plot
    colors_mae = plt.cm.Reds(metrics_df['MAE'] / metrics_df['MAE'].max())
    ax1.barh(metrics_df['Target'], metrics_df['MAE'], color=colors_mae)
    ax1.set_xlabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance - MAE', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # R² plot
    colors_r2 = plt.cm.Greens(metrics_df['R²'])
    ax2.barh(metrics_df['Target'], metrics_df['R²'], color=colors_r2)
    ax2.set_xlabel('R² Score', fontsize=12, fontweight='bold')
    ax2.set_title('Model Performance - R² Score', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, 'xgboost_performance_summary.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(FIGURES_DIR, 'xgboost_performance_summary.png')}")
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("XGBoost Model Visualization")
    print("="*70)
    
    print("\n1. Creating Prediction vs Actual plots...")
    plot_prediction_vs_actual()
    
    print("\n2. Creating Feature Importance plots...")
    plot_feature_importance()
    
    print("\n3. Creating Error Distribution plots...")
    plot_error_distribution()
    
    print("\n4. Creating Performance Summary plot...")
    plot_performance_summary()
    
    print("\n" + "="*70)
    print("All visualizations created successfully!")
    print("Check the 'figures/' directory for output files.")
    print("="*70)
