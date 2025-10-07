"""
Refined example script for using the trained XGBoost body measurement predictor.

This script demonstrates the HYBRID PREDICTION METHODOLOGY:
1.  It takes the direct measurements from a CNN (simulated here).
2.  It feeds these measurements into the trained XGBoost model to get a holistic prediction.
3.  It intelligently combines the results, averaging the CNN's direct measurement with
    the XGBoost model's holistic prediction for overlapping measurements.
"""

import pandas as pd
import numpy as np
import os
import glob
# Ensure this import points to your training script file where BodyMeasurementPredictor is defined
from train_predictor_model import BodyMeasurementPredictor


def get_latest_model_path():
    """
    Get the path to the most recently saved model.
    """
    # Create a path to the model directory relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, '..', 'data', 'model_files')
    
    # Find all XGBoost model files
    search_path = os.path.join(model_dir, 'xgboost_body_predictor_*.pkl')
    model_files = glob.glob(search_path)
    
    if not model_files:
        raise FileNotFoundError(f"No XGBoost models found in {model_dir}")
    
    # Return the most recent one
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model


def predict_single_person_refined():
    """
    Example: Predict measurements for a single person using the refined hybrid method.
    """
    print("\n" + "="*70)
    print("Refined Single Person Prediction Example")
    print("="*70)
    
    # Initialize predictor
    predictor = BodyMeasurementPredictor()
    
    # Load the retrained model (which expects the extra proxy measurements)
    model_path = get_latest_model_path()
    predictor.load_model(model_path)
    
    # --- STEP 1: Get outputs from your (simulated) CNN ---
    cnn_direct_extractions = {
        'Stature': 1750,
        'Weight': 75.0,
        'Chest_Circumference': 950,
        'Hip_Circumference': 980,
        'Waist_Circumference': 850,
        'Thigh_Circumference': 580,
        'Ankle_Circumference': 230,
        'Shoulder_Breadth': 420,
        'Knee_Height': 500,
        'Arm_Circumference_Flexed': 345, # CNN's direct measurement
        'Calf_Circumference': 380,     # CNN's direct measurement
        'BMI': 24.5,
        'Age': 30,
        'Gender_Male': 1
    }
    
    print("\n--- STEP 1: CNN Direct Extractions (Input) ---")
    for key, value in cnn_direct_extractions.items():
        print(f"  {key:<25}: {value}")

    # --- STEP 2: Use the XGBoost model for holistic prediction ---
    # IMPORTANT: Create DataFrame with columns in the exact order as proxy_measurements
    X_input = pd.DataFrame([cnn_direct_extractions])
    # Reorder columns to match the model's expected order
    X_input = X_input[predictor.proxy_measurements]
    
    # Get the holistic predictions from the XGBoost model
    xgb_predictions_array = predictor.predict(X_input)
    xgb_predictions = pd.DataFrame(xgb_predictions_array, columns=predictor.target_measurements)

    print("\n--- STEP 2: XGBoost Holistic Predictions ---")
    print(xgb_predictions.round(2).to_string(index=False))

    # --- STEP 3: Combine the results for the final measurements ---
    final_measurements = xgb_predictions.copy()

    # For overlapping measurements, average the CNN's direct view with the XGBoost's holistic view
    cnn_arm_circ = cnn_direct_extractions['Arm_Circumference_Flexed']
    xgb_arm_circ = xgb_predictions['Arm_Circumference_Flexed'].iloc[0]
    final_measurements['Arm_Circumference_Flexed'] = (cnn_arm_circ + xgb_arm_circ) / 2

    cnn_calf_circ = cnn_direct_extractions['Calf_Circumference']
    xgb_calf_circ = xgb_predictions['Calf_Circumference'].iloc[0]
    final_measurements['Calf_Circumference'] = (cnn_calf_circ + xgb_calf_circ) / 2

    print("\n--- STEP 3: Final Refined Measurements (Averaged) ---")
    print(final_measurements.round(2).to_string(index=False))
    
    return final_measurements


def analyze_feature_importance():
    """
    Analyze which input features are most important for predictions.
    This function remains the same as it inspects the trained model properties.
    """
    print("\n" + "="*70)
    print("Feature Importance Analysis")
    print("="*70)
    
    predictor = BodyMeasurementPredictor()
    model_path = get_latest_model_path()
    predictor.load_model(model_path)
    
    print("\nTop 5 most important features for each target measurement:\n")
    for i, target in enumerate(predictor.target_measurements):
        print(f"--- Predicting: {target} ---")
        estimator = predictor.model.estimators_[i]
        importances = estimator.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': predictor.proxy_measurements,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print(importance_df.head(5).to_string(index=False))
        print()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("XGBoost Predictor - Refined Usage Example")
    print("="*70)
    
    # Run the refined single person prediction example
    predict_single_person_refined()
    
    # Run the feature importance analysis on the retrained model
    analyze_feature_importance()
    
    print("\n" + "="*70)
    print("Examples Complete!")
    print("="*70)
