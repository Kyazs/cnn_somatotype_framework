#!/usr/bin/env python3
"""
CAESAR Somatotype Model Training Script

This script will:
1. Check what CAESAR data is actually available
2. Create proper training/validation datasets
3. Train real models on actual anthropometric data
4. Evaluate model performance with proper metrics
5. Save trained models for production use

Author: AI Assistant
Date: September 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import DS_DIR, MODEL_FILES_DIR

class SomatotypeModelTrainer:
    """
    Professional training system for somatotype prediction models.
    
    This class will:
    - Audit available CAESAR data
    - Train models only on real anthropometric measurements
    - Provide realistic performance metrics
    - Save properly validated models
    """
    
    def __init__(self, gender="male"):
        self.gender = gender
        self.caesar_data = None
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.training_log = []
        
        # Define the 8 target somatotype measurements we want to predict
        self.somatotype_targets = [
            'triceps_skinfold_mm',
            'subscapular_skinfold_mm', 
            'suprailiac_skinfold_mm',
            'calf_skinfold_mm',
            'humerus_biepicondylar_breadth_cm',
            'femur_biepicondylar_breadth_cm',
            'arm_circumference_flexed_cm',
            'calf_circumference_cm'
        ]
        
        # Common CAESAR column name variations
        self.caesar_column_mapping = {
            'triceps_skinfold_mm': [
                'Triceps Skinfold', 'triceps_skinfold', 'Tricep Skinfold', 
                'triceps_skinfold_mm', 'TRICEPS_SKINFOLD'
            ],
            'subscapular_skinfold_mm': [
                'Subscapular Skinfold', 'subscapular_skinfold', 'Subscap Skinfold',
                'subscapular_skinfold_mm', 'SUBSCAPULAR_SKINFOLD'
            ],
            'suprailiac_skinfold_mm': [
                'Suprailiac Skinfold', 'suprailiac_skinfold', 'Suprailium Skinfold',
                'suprailiac_skinfold_mm', 'SUPRAILIAC_SKINFOLD'
            ],
            'calf_skinfold_mm': [
                'Calf Skinfold', 'calf_skinfold', 'Medial Calf Skinfold',
                'calf_skinfold_mm', 'CALF_SKINFOLD'
            ],
            'humerus_biepicondylar_breadth_cm': [
                'Elbow Breadth', 'humerus_breadth', 'Biepicondylar Humerus',
                'Humerus Biepicondylar Breadth', 'elbow_breadth'
            ],
            'femur_biepicondylar_breadth_cm': [
                'Knee Breadth', 'femur_breadth', 'Biepicondylar Femur',
                'Femur Biepicondylar Breadth', 'knee_breadth'
            ],
            'arm_circumference_flexed_cm': [
                'Arm Circumference Flexed', 'arm_circ_flexed', 'Bicep Circumference',
                'Arm Circumference, Flexed', 'arm_circumference_flexed'
            ],
            'calf_circumference_cm': [
                'Calf Circumference', 'calf_circ', 'Calf Maximum Girth',
                'Calf Circumference, Maximum', 'calf_circumference'
            ]
        }
        
        # Potential predictor features in CAESAR
        self.potential_predictors = [
            'Stature', 'Weight', 'Age',
            'Chest Circumference', 'Waist Circumference', 'Hip Circumference',
            'Neck Circumference', 'Thigh Circumference', 'Ankle Circumference',
            'Shoulder Breadth', 'Hip Breadth', 'Chest Depth', 'Waist Depth',
            'Armscye Circumference', 'Arm Length', 'Forearm Length',
            'Hand Length', 'Foot Length', 'Crotch Height', 'Knee Height',
            'Sitting Height', 'Eye Height', 'Shoulder Height'
        ]
    
    def audit_available_data(self):
        """
        Comprehensive audit of what CAESAR data is actually available.
        This is crucial to determine if we can train real models.
        """
        print("=" * 70)
        print("AUDITING AVAILABLE CAESAR DATA")
        print("=" * 70)
        
        # Check all possible CAESAR file locations
        possible_files = [
            f"measurements_CAESAR_{self.gender}_somatotype.csv",
            # f"caesar_{self.gender}_somatotype.csv", 
            # f"caesar_{self.gender}.csv",
            # f"measurements_CAESAR_{self.gender}.csv",
            # f"CAESAR_{self.gender}.csv",
            # f"{self.gender}_caesar.csv",
            # f"caesar_data_{self.gender}.csv"
        ]
        
        print(f"Searching for CAESAR data files in: {DS_DIR}")
        caesar_file = None
        
        for filename in possible_files:
            filepath = os.path.join(DS_DIR, filename)
            if os.path.exists(filepath):
                caesar_file = filepath
                print(f"Found CAESAR data: {filename}")
                break
            else:
                print(f"Not found: {filename}")
        
        if not caesar_file:
            print("\nCRITICAL ERROR: No CAESAR dataset found!")
            print("Available files in datasets directory:")
            if os.path.exists(DS_DIR):
                for file in os.listdir(DS_DIR):
                    if file.endswith('.csv'):
                        print(f"  - {file}")
            else:
                print(f"Directory {DS_DIR} does not exist!")
            return False
        
        # Load and examine the dataset
        print(f"\nLoading dataset: {os.path.basename(caesar_file)}")
        self.caesar_data = pd.read_csv(caesar_file)
        
        print(f"Dataset shape: {self.caesar_data.shape}")
        print(f"Number of subjects: {len(self.caesar_data)}")
        print(f"Number of measurements: {len(self.caesar_data.columns)}")
        
        # Show sample columns
        print(f"\nSample columns (first 20):")
        for i, col in enumerate(self.caesar_data.columns[:20]):
            print(f"  {i+1:2d}. {col}")
        if len(self.caesar_data.columns) > 20:
            print(f"  ... and {len(self.caesar_data.columns) - 20} more columns")
        
        return True
    
    def check_somatotype_availability(self):
        """
        Check which of the 8 somatotype measurements are actually available in CAESAR data.
        """
        print("\n" + "=" * 70)
        print("CHECKING SOMATOTYPE MEASUREMENT AVAILABILITY")
        print("=" * 70)
        
        self.available_targets = {}
        self.missing_targets = []
        self.available_predictors = []
        
        # Check somatotype targets
        print("\nSOMATOTYPE MEASUREMENTS:")
        print("-" * 50)
        
        for target in self.somatotype_targets:
            found_column = None
            
            # Check all possible column name variations
            for alt_name in self.caesar_column_mapping[target]:
                if alt_name in self.caesar_data.columns:
                    found_column = alt_name
                    break
            
            if found_column:
                non_null_count = self.caesar_data[found_column].notna().sum()
                if non_null_count >= 50:  # Minimum samples for training
                    self.available_targets[target] = found_column
                    mean_val = self.caesar_data[found_column].mean()
                    std_val = self.caesar_data[found_column].std()
                    min_val = self.caesar_data[found_column].min()
                    max_val = self.caesar_data[found_column].max()
                    
                    print(f"[OK] {target}")
                    print(f"   Column: '{found_column}'")
                    print(f"   Samples: {non_null_count}")
                    print(f"   Range: {min_val:.2f} - {max_val:.2f} (μ={mean_val:.2f}, σ={std_val:.2f})")
                else:
                    self.missing_targets.append(target)
                    print(f"[WARN] {target}")
                    print(f"   Column: '{found_column}' (only {non_null_count} samples - insufficient)")
            else:
                self.missing_targets.append(target)
                print(f"[MISS] {target}")
                print(f"   No matching column found")
            print()
        
        # Check predictor availability
        print("\nPREDICTOR AVAILABILITY:")
        print("-" * 50)
        
        for predictor in self.potential_predictors:
            if predictor in self.caesar_data.columns:
                non_null_count = self.caesar_data[predictor].notna().sum()
                if non_null_count >= 100:  # Need good coverage for predictors
                    self.available_predictors.append(predictor)
                    print(f"[OK] {predictor}: {non_null_count} samples")
                else:
                    print(f"[WARN] {predictor}: {non_null_count} samples (insufficient)")
            else:
                # Check variations
                variations = [
                    predictor.replace(' ', '_'), 
                    predictor.replace(' ', '').upper(),
                    predictor.replace(' ', '').lower()
                ]
                found = False
                for var in variations:
                    if var in self.caesar_data.columns:
                        self.available_predictors.append(var)
                        print(f"[OK] {predictor} (as '{var}')")
                        found = True
                        break
                if not found:
                    print(f"[MISS] {predictor}")
        
        # Summary
        print("\n" + "=" * 70)
        print("DATA AVAILABILITY SUMMARY")
        print("=" * 70)
        print(f"Somatotype measurements available: {len(self.available_targets)}/8")
        print(f"Predictor features available: {len(self.available_predictors)}")
        print(f"Total dataset size: {len(self.caesar_data)} subjects")
        
        if len(self.available_targets) == 0:
            print("\nCRITICAL ISSUE: No somatotype measurements found!")
            print("The current system cannot train real models.")
            return False
        elif len(self.available_targets) < 4:
            print("\nWARNING: Limited somatotype data available.")
            print("Models will only be trained for available measurements.")
        else:
            print("\nGood data availability for model training!")
        
        return len(self.available_targets) > 0
    
    def create_training_plan(self):
        """Create a comprehensive training plan based on available data."""
        print("\n" + "=" * 70)
        print("CREATING TRAINING PLAN")
        print("=" * 70)
        
        self.training_plan = {}
        
        for target, column in self.available_targets.items():
            # Identify best predictors for this target
            target_data = self.caesar_data[column].dropna()
            
            # Calculate correlations with available predictors
            correlations = {}
            for predictor in self.available_predictors:
                if predictor in self.caesar_data.columns:
                    corr = self.caesar_data[column].corr(self.caesar_data[predictor])
                    if not np.isnan(corr):
                        correlations[predictor] = abs(corr)
            
            # Select top correlated predictors
            top_predictors = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:8]
            selected_predictors = [p[0] for p in top_predictors if p[1] > 0.1]  # Minimum correlation
            
            self.training_plan[target] = {
                'target_column': column,
                'predictors': selected_predictors,
                'correlations': dict(top_predictors),
                'sample_size': len(target_data),
                'target_stats': {
                    'mean': target_data.mean(),
                    'std': target_data.std(),
                    'min': target_data.min(),
                    'max': target_data.max()
                }
            }
            
            print(f"\n[TARGET] {target}")
            print(f"   Target column: {column}")
            print(f"   Sample size: {len(target_data)}")
            print(f"   Selected predictors: {len(selected_predictors)}")
            for pred, corr in top_predictors[:5]:  # Show top 5 correlations
                print(f"     - {pred}: r={corr:.3f}")
        
        return len(self.training_plan) > 0
    
    def train_models(self):
        """Train actual models on real CAESAR data."""
        print("\n" + "=" * 70)
        print("TRAINING SOMATOTYPE MODELS")
        print("=" * 70)
        
        if not self.training_plan:
            print("No training plan available!")
            return False
        
        successful_models = 0
        
        for target, plan in self.training_plan.items():
            print(f"\nTraining model for: {target}")
            print("-" * 50)
            
            try:
                success = self._train_single_model(target, plan)
                if success:
                    successful_models += 1
                    print(f"Successfully trained {target}")
                else:
                    print(f"Failed to train {target}")
            except Exception as e:
                print(f"Error training {target}: {e}")
        
        print(f"\nTRAINING RESULTS:")
        print(f"Successfully trained: {successful_models}/{len(self.training_plan)} models")
        
        return successful_models > 0
    
    def _train_single_model(self, target, plan):
        """Train a single model for one somatotype measurement."""
        target_column = plan['target_column']
        predictors = plan['predictors']
        
        if len(predictors) < 2:
            print(f"Insufficient predictors for {target}")
            return False
        
        # Prepare data
        feature_cols = predictors + [target_column]
        model_data = self.caesar_data[feature_cols].dropna()
        
        if len(model_data) < 50:
            print(f"Insufficient samples after cleaning: {len(model_data)}")
            return False
        
        X = model_data[predictors]
        y = model_data[target_column]
        
        print(f"   Data shape: X={X.shape}, y={y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Try different models
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Ridge': Ridge(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=0.1, random_state=42)
        }
        
        best_model = None
        best_score = -np.inf
        best_name = None
        
        print("   Testing model types:")
        for name, model in models.items():
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            mean_score = cv_scores.mean()
            print(f"     {name}: CV R² = {mean_score:.3f} (±{cv_scores.std():.3f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
                best_name = name
        
        # Train best model
        best_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = best_model.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"   Best model: {best_name}")
        print(f"   Test R²: {test_r2:.3f}")
        print(f"   Test MAE: {test_mae:.3f}")
        print(f"   Test RMSE: {test_rmse:.3f}")
        
        # Store model
        self.models[target] = {
            'model': best_model,
            'scaler': scaler,
            'predictors': predictors,
            'model_type': best_name
        }
        
        # Store performance
        self.model_performance[target] = {
            'model_type': best_name,
            'cv_r2': best_score,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'n_samples': len(model_data),
            'n_features': len(predictors),
            'predictors': predictors
        }
        
        return True
    
    def save_models(self):
        """Save trained models and metadata."""
        print("\n" + "=" * 70)
        print("SAVING TRAINED MODELS")
        print("=" * 70)
        
        model_dir = os.path.join(MODEL_FILES_DIR, "somatotype_models")
        os.makedirs(model_dir, exist_ok=True)
        
        saved_count = 0
        for target, model_info in self.models.items():
            try:
                # Save model
                model_file = os.path.join(model_dir, f"{target}_{self.gender}_model.pkl")
                joblib.dump(model_info, model_file)
                print(f"Saved: {target}")
                saved_count += 1
            except Exception as e:
                print(f"Failed to save {target}: {e}")
        
        # Save performance report
        report_file = os.path.join(model_dir, f"training_report_{self.gender}.json")
        report = {
            'training_date': datetime.now().isoformat(),
            'gender': self.gender,
            'dataset_info': {
                'total_subjects': len(self.caesar_data),
                'available_targets': len(self.available_targets),
                'available_predictors': len(self.available_predictors)
            },
            'model_performance': self.model_performance,
            'training_plan': self.training_plan
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nSAVE SUMMARY:")
        print(f"Models saved: {saved_count}/{len(self.models)}")
        print(f"Report saved: {report_file}")
        
        return saved_count > 0
    
    def run_complete_training(self):
        """Run the complete training pipeline."""
        print("STARTING SOMATOTYPE MODEL TRAINING")
        print("=" * 70)
        
        # Step 1: Audit data
        if not self.audit_available_data():
            print("Training aborted: No data available")
            return False
        
        # Step 2: Check somatotype availability
        if not self.check_somatotype_availability():
            print("Training aborted: No somatotype measurements available")
            return False
        
        # Step 3: Create training plan
        if not self.create_training_plan():
            print("Training aborted: Cannot create training plan")
            return False
        
        # Step 4: Train models
        if not self.train_models():
            print("Training failed: No models successfully trained")
            return False
        
        # Step 5: Save models
        if not self.save_models():
            print("Failed to save models")
            return False
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Trained models for {len(self.models)} somatotype measurements")
        print(f"Gender: {self.gender}")
        print(f"Dataset size: {len(self.caesar_data)} subjects")
        
        return True


def main():
    """Main function to run somatotype model training."""
    print("CAESAR SOMATOTYPE MODEL TRAINER")
    print("=" * 70)
    print("This script will audit available data and train real models")
    print("for somatotype measurements using the CAESAR dataset.")
    print("=" * 70)
    
    # Train for both genders
    for gender in ['male', 'female']:
        print(f"\nTRAINING MODELS FOR: {gender.upper()}")
        print("=" * 70)
        
        trainer = SomatotypeModelTrainer(gender=gender)
        success = trainer.run_complete_training()
        
        if success:
            print(f"{gender.title()} models trained successfully!")
        else:
            print(f"{gender.title()} model training failed!")
        
        print("\n" + "=" * 70)
    
    print("\nTRAINING PIPELINE COMPLETED!")


if __name__ == "__main__":
    main()
