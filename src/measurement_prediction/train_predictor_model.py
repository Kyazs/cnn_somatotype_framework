"""
XGBoost Regression Model for Body Measurement Prediction
=========================================================
This script uses XGBoost to predict target body measurements from proxy measurements
using the CAESAR dataset. The model uses gradient boosting to sequentially learn from
prediction errors, creating a powerful ensemble of decision trees.
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os
import sys
import pickle
from datetime import datetime

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import MODEL_FILES_DIR


class BodyMeasurementPredictor:
    """
    XGBoost-based regression model for predicting body measurements.

    This model uses a multi-output regression approach to predict multiple
    target measurements simultaneously from a set of proxy measurements.
    """

    def __init__(self, model_save_dir=None):
        """
        Initialize the predictor.

        Args:
            model_save_dir: Directory to save trained models
        """
        if model_save_dir is None:
            # Use MODEL_FILES_DIR from utils.py
            model_save_dir = MODEL_FILES_DIR

        self.model_save_dir = model_save_dir
        self.model = None
        self.proxy_measurements = None
        self.target_measurements = None
        self.scaler_X = None
        self.scaler_y = None

    def define_measurements(self):
        """
        Define proxy (input) and target (output) measurements based on the CAESAR dataset.
        
        Note: Arm_Circumference_Flexed and Calf_Circumference appear in both proxy and target.
        This is intentional - they come from CNN predictions (proxy) and are refined by 
        XGBoost regression (target) for improved accuracy in a cascaded prediction approach.
        """
        # Proxy measurements - easily obtainable measurements + CNN predictions
        self.proxy_measurements = [
            "Stature",  # Height
            "Weight",  # Body weight
            "Chest_Circumference",  # Chest circumference
            "Hip_Circumference",  # Hip circumference
            "Waist_Circumference",  # Waist circumference
            "Thigh_Circumference",  # Thigh circumference
            "Ankle_Circumference",  # Ankle circumference
            "Shoulder_Breadth",  # Shoulder width
            "Knee_Height",  # Knee height
            "Arm_Circumference_Flexed",  # From CNN model (to be refined)
            "Calf_Circumference",  # From CNN model (to be refined)
            "BMI",  # Body Mass Index
            "Age",  # Age
            "Gender_Male",  # Gender (1 for male, 0 for female)
        ]

        # Target measurements - measurements to predict/refine with XGBoost
        self.target_measurements = [
            "Triceps_Skinfold",  # Skinfold measurement at triceps
            "Subscapular_Skinfold",  # Skinfold at subscapular area
            "Supraspinale_Skinfold",  # Skinfold at supraspinale
            "Calf_Skinfold",  # Skinfold at calf
            "Humerus_Breadth",  # Humerus bone breadth
            "Femur_Breadth",  # Femur bone breadth
            "Arm_Circumference_Flexed",  # Refined prediction from CNN input
            "Calf_Circumference",  # Refined prediction from CNN input
        ]

    def load_data(self, csv_path):
        """
        Load the CAESAR dataset from CSV file.

        Args:
            csv_path: Path to the CSV file

        Returns:
            DataFrame containing the dataset
        """
        print(f"\n{'='*70}")
        print("Loading CAESAR Dataset")
        print(f"{'='*70}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found at: {csv_path}")

        df = pd.read_csv(csv_path)
        print(f"Dataset loaded successfully!")
        print(f"Total samples: {len(df)}")
        print(f"Total features: {len(df.columns)}")

        return df

    def prepare_data(self, df):
        """
        Prepare features and targets from the dataset.

        Args:
            df: DataFrame containing the dataset

        Returns:
            X, y: Features and targets
        """
        print(f"\n{'='*70}")
        print("Preparing Data")
        print(f"{'='*70}")

        # Ensure all selected columns exist
        missing_cols = [
            col
            for col in self.proxy_measurements + self.target_measurements
            if col not in df.columns
        ]
        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")

        # Extract features and targets
        X = df[self.proxy_measurements].copy()
        y = df[self.target_measurements].copy()

        # Handle any missing values
        print(f"\nChecking for missing values...")
        missing_X = X.isnull().sum().sum()
        missing_y = y.isnull().sum().sum()

        if missing_X > 0:
            print(
                f"Warning: {missing_X} missing values found in features. Filling with median."
            )
            X = X.fillna(X.median())

        if missing_y > 0:
            print(
                f"Warning: {missing_y} missing values found in targets. Filling with median."
            )
            y = y.fillna(y.median())

        print(f"\nFeatures shape: {X.shape}")
        print(f"Targets shape: {y.shape}")
        print(f"\nProxy measurements ({len(self.proxy_measurements)}):")
        for i, col in enumerate(self.proxy_measurements, 1):
            print(f"  {i}. {col}")

        print(f"\nTarget measurements ({len(self.target_measurements)}):")
        for i, col in enumerate(self.target_measurements, 1):
            print(f"  {i}. {col}")

        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.

        Args:
            X: Features
            y: Targets
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            X_train, X_test, y_train, y_test
        """
        print(f"\n{'='*70}")
        print("Splitting Data")
        print(f"{'='*70}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"Training set: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
        print(f"Testing set: {X_test.shape[0]} samples ({test_size*100:.0f}%)")

        return X_train, X_test, y_train, y_test

    def build_model(self):
        """
        Build the XGBoost multi-output regression model.

        Returns:
            MultiOutputRegressor wrapping XGBRegressor
        """
        print(f"\n{'='*70}")
        print("Building XGBoost Model")
        print(f"{'='*70}")

        # XGBoost parameters optimized for body measurement prediction
        xgb_regressor = xgb.XGBRegressor(
            objective="reg:squarederror",  # Squared error for regression
            n_estimators=1000,  # Number of boosting rounds (trees)
            learning_rate=0.05,  # Step size shrinkage (eta)
            max_depth=6,  # Maximum depth of trees
            subsample=0.8,  # Fraction of samples per tree
            colsample_bytree=0.8,  # Fraction of features per tree
            min_child_weight=1,  # Minimum sum of instance weight in a child
            gamma=0,  # Minimum loss reduction for split
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
            eval_metric="rmse",  # Evaluation metric
        )

        # Wrap in MultiOutputRegressor for handling multiple targets
        self.model = MultiOutputRegressor(estimator=xgb_regressor)

        print("Model architecture:")
        print(f"  - Base estimator: XGBRegressor")
        print(f"  - Number of estimators: 1000")
        print(f"  - Learning rate: 0.05")
        print(f"  - Max depth: 6")
        print(f"  - Subsample ratio: 0.8")
        print(f"  - Feature sampling ratio: 0.8")

        return self.model

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the XGBoost model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        print(f"\n{'='*70}")
        print("Training Model")
        print(f"{'='*70}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if self.model is None:
            self.build_model()

        # Train the model
        print("\nTraining in progress...")
        print("(This may take several minutes depending on dataset size)")

        if X_val is not None and y_val is not None:
            # Train with validation set for early stopping
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        else:
            self.model.fit(X_train, y_train)

        print(f"\nTraining completed!")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"\n{'='*70}")
        print("Model Evaluation")
        print(f"{'='*70}")

        # Make predictions
        print("Generating predictions...")
        y_pred = self.model.predict(X_test)

        # Calculate metrics for each target
        results = {}

        print(f"\n{'Measurement':<30} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
        print("-" * 70)

        mae_list = []
        rmse_list = []
        r2_list = []

        for i, target_name in enumerate(self.target_measurements):
            mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
            r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])

            mae_list.append(mae)
            rmse_list.append(rmse)
            r2_list.append(r2)

            results[target_name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

            print(f"{target_name:<30} {mae:<10.3f} {rmse:<10.3f} {r2:<10.3f}")

        # Overall metrics
        print("-" * 70)
        avg_mae = np.mean(mae_list)
        avg_rmse = np.mean(rmse_list)
        avg_r2 = np.mean(r2_list)

        print(f"{'AVERAGE':<30} {avg_mae:<10.3f} {avg_rmse:<10.3f} {avg_r2:<10.3f}")

        results["overall"] = {"MAE": avg_mae, "RMSE": avg_rmse, "R2": avg_r2}

        print(f"\n{'='*70}")
        print("Evaluation Summary")
        print(f"{'='*70}")
        print(f"Average Mean Absolute Error (MAE): {avg_mae:.3f}")
        print(f"Average Root Mean Squared Error (RMSE): {avg_rmse:.3f}")
        print(f"Average R² Score: {avg_r2:.3f}")
        print(f"\nInterpretation:")
        print(f"  - MAE: On average, predictions are off by {avg_mae:.2f} units")
        print(f"  - R² Score: The model explains {avg_r2*100:.1f}% of the variance")

        return results

    def save_model(self, filename=None):
        """
        Save the trained model to disk.

        Args:
            filename: Name of the file to save (without extension)
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        os.makedirs(self.model_save_dir, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"xgboost_body_predictor_{timestamp}"

        filepath = os.path.join(self.model_save_dir, f"{filename}.pkl")

        # Save model and metadata
        model_data = {
            "model": self.model,
            "proxy_measurements": self.proxy_measurements,
            "target_measurements": self.target_measurements,
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"\n{'='*70}")
        print(f"Model saved to: {filepath}")
        print(f"{'='*70}")

    def load_model(self, filepath):
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.proxy_measurements = model_data["proxy_measurements"]
        self.target_measurements = model_data["target_measurements"]

        print(f"\n{'='*70}")
        print(f"Model loaded from: {filepath}")
        print(f"Trained on: {model_data.get('timestamp', 'Unknown')}")
        print(f"{'='*70}")

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X: Features (can be DataFrame or array)

        Returns:
            Predictions for all target measurements
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")

        return self.model.predict(X)


def main():
    """
    Main function to train and evaluate the XGBoost body measurement predictor.
    """
    print(f"\n{'='*70}")
    print("XGBoost Body Measurement Prediction System")
    print(f"{'='*70}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize predictor
    predictor = BodyMeasurementPredictor()

    # Define measurements
    predictor.define_measurements()

    # Load data
    dataset_path = r"C:\Users\LENOVO\Desktop\Kekious_Maximus\human-body-reshape-DL-paper\data\datasets\CAESAR_full_imputation_dataset.csv"
    df = predictor.load_data(dataset_path)

    # Prepare data
    X, y = predictor.prepare_data(df)

    # Split data
    X_train, X_test, y_train, y_test = predictor.split_data(X, y, test_size=0.2)

    # Build and train model
    predictor.build_model()
    predictor.train(X_train, y_train)

    # Evaluate model
    results = predictor.evaluate(X_test, y_test)

    # Save model
    predictor.save_model()

    print(f"\n{'='*70}")
    print("Process Complete!")
    print(f"{'='*70}")

    return predictor, results


if __name__ == "__main__":
    predictor, results = main()
