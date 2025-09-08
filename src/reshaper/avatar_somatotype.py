import numpy as np
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import joblib

# Import the original Avatar class
from .avatar import Avatar

# Add parent directory to path to import utils
sys.path.append("..")
from utils import *


class AvatarSomatotype(Avatar):
    """
    Enhanced Avatar class with CAESAR-based somatotype measurement capabilities.
    
    This class extends the original Avatar class to predict 8 specialized somatotype 
    measurements using the CAESAR dataset:
    
    Skinfold Measurements (4):
    - Triceps skinfold (mm)
    - Subscapular skinfold (mm) 
    - Suprailiac skinfold (mm)
    - Calf skinfold (mm)
    
    Bone Breadth Measurements (2):
    - Humerus biepicondylar breadth (cm)
    - Femur biepicondylar breadth (cm)
    
    Specialized Circumferences (2):
    - Arm circumference, flexed and tensed (cm)
    - Calf circumference (cm)
    """
    
    # Define somatotype measurement names
    SOMATOTYPE_MEASUREMENTS = [
        'triceps_skinfold_mm',
        'subscapular_skinfold_mm', 
        'suprailiac_skinfold_mm',
        'calf_skinfold_mm',
        'humerus_biepicondylar_breadth_cm',
        'femur_biepicondylar_breadth_cm',
        'arm_circumference_flexed_cm',
        'calf_circumference_cm'
    ]
    
    def __init__(self, input_data, gender="female", enable_somatotype=True):
        """
        Initialize the enhanced Avatar with somatotype capabilities.
        
        Args:
            input_data (numpy ndarray): input measurements of size 21
            gender (str): "female" or "male"
            enable_somatotype (bool): whether to enable somatotype predictions
        """
        # Initialize parent Avatar class
        super().__init__(input_data, gender)
        
        # Somatotype-specific attributes
        self.enable_somatotype = enable_somatotype
        self.somatotype_measurements = {}
        self.somatotype_models = {}
        self.caesar_data = None
        self.somatotype_confidence = {}
        
        # Initialize somatotype system if enabled
        if self.enable_somatotype:
            self._initialize_somatotype_system()
    
    def _initialize_somatotype_system(self):
        """Initialize the CAESAR-based somatotype prediction system"""
        print("Initializing CAESAR-based somatotype system...")
        
        try:
            # Load properly trained models and their performance metrics
            self._load_trained_models()
            print("Somatotype system initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize somatotype system: {e}")
            self.enable_somatotype = False
    
    def _load_trained_models(self):
        """Load properly trained somatotype models with real performance metrics"""
        model_dir = os.path.join(MODEL_FILES_DIR, "somatotype_models")
        
        # Load training report with real performance metrics
        report_file = os.path.join(model_dir, f"training_report_{self.gender}.json")
        if os.path.exists(report_file):
            import json
            with open(report_file, 'r') as f:
                self.training_report = json.load(f)
            print(f"Loaded training report: {len(self.training_report.get('model_performance', {}))} models")
        else:
            print(f"Warning: Training report not found: {report_file}")
            self.training_report = {}
        
        # Load individual model files
        loaded_models = 0
        for measurement in self.SOMATOTYPE_MEASUREMENTS:
            model_file = os.path.join(model_dir, f"{measurement}_{self.gender}_model.pkl")
            
            if os.path.exists(model_file):
                try:
                    model_data = joblib.load(model_file)
                    self.somatotype_models[measurement] = model_data
                    loaded_models += 1
                    print(f"✓ Loaded model for {measurement}")
                except Exception as e:
                    print(f"✗ Failed to load {measurement}: {e}")
            else:
                print(f"✗ Model file not found: {measurement}")
        
        if loaded_models == 0:
            raise FileNotFoundError("No trained somatotype models found")
        
        print(f"Successfully loaded {loaded_models}/{len(self.SOMATOTYPE_MEASUREMENTS)} somatotype models")
    
    def _get_model_confidence(self, measurement):
        """Get realistic confidence score based on actual model performance"""
        if measurement not in self.training_report.get('model_performance', {}):
            return 0.3  # Low confidence for missing models
        
        performance = self.training_report['model_performance'][measurement]
        
        # Calculate confidence based on R² score and sample size
        r2_score = performance.get('test_r2', 0)
        n_samples = performance.get('n_samples', 0)
        n_features = performance.get('n_features', 0)
        
        # Base confidence from R² score (0-1 scale)
        base_confidence = max(0, min(1, r2_score))
        
        # Adjust for sample size (more samples = higher confidence)
        sample_adjustment = min(0.2, n_samples / 5000)  # Up to 0.2 bonus for large samples
        
        # Adjust for feature count (more features can improve prediction)
        feature_adjustment = min(0.1, n_features / 20)  # Up to 0.1 bonus for many features
        
        # Penalize very low R² scores
        if r2_score < 0.2:
            base_confidence *= 0.5
        
        final_confidence = base_confidence + sample_adjustment + feature_adjustment
        return round(min(0.95, max(0.1, final_confidence)), 3)
    
    def _get_prediction_uncertainty(self, measurement):
        """Get prediction uncertainty based on model performance"""
        if measurement not in self.training_report.get('model_performance', {}):
            return {'mae': 10.0, 'rmse': 15.0}  # High uncertainty for missing models
        
        performance = self.training_report['model_performance'][measurement]
        return {
            'mae': performance.get('test_mae', 5.0),
            'rmse': performance.get('test_rmse', 8.0)
        }
    
    def predict_complete(self, db_name="TOTAL", include_somatotype=True):
        """
        Predict complete anthropometric measurements including somatotype.
        
        Args:
            db_name (str): Database name for base predictions
            include_somatotype (bool): Whether to include somatotype predictions
            
        Returns:
            dict: Complete measurement results with confidence scores
        """
        # Get base measurements from parent class
        base_measurements = self.predict(db_name)
        
        results = {
            'base_measurements': base_measurements,
            'somatotype_measurements': {},
            'confidence_scores': {},
            'validation_status': {'status': 'success', 'warnings': [], 'errors': []}
        }
        
        # Add somatotype predictions if enabled
        if include_somatotype and self.enable_somatotype:
            try:
                somatotype_results = self.predict_somatotype_measurements()
                results['somatotype_measurements'] = somatotype_results['measurements']
                results['confidence_scores'] = somatotype_results['confidence']
                
                # Validate measurements
                validation = self._validate_measurements(results['somatotype_measurements'])
                results['validation_status'].update(validation)
                
            except Exception as e:
                results['validation_status']['errors'].append(f"Somatotype prediction failed: {str(e)}")
        
        return results
    
    def predict_somatotype_measurements(self):
        """Predict all somatotype measurements"""
        measurements = {}
        confidence_scores = {}
        
        # Convert current measurements to input format for CAESAR models
        input_features = self._prepare_input_features()
        
        for measurement in self.SOMATOTYPE_MEASUREMENTS:
            if measurement in self.somatotype_models:
                try:
                    pred_result = self._predict_single_measurement(measurement, input_features)
                    measurements[measurement] = pred_result['value']
                    confidence_scores[measurement] = pred_result['confidence']
                except Exception as e:
                    print(f"Failed to predict {measurement}: {e}")
                    measurements[measurement] = None
                    confidence_scores[measurement] = 0.0
            else:
                measurements[measurement] = None
                confidence_scores[measurement] = 0.0
        
        return {
            'measurements': measurements,
            'confidence': confidence_scores
        }
    
    def _prepare_input_features(self):
        """Prepare input features from current measurements for trained CAESAR models"""
        # Helper function to safely extract scalar values
        def safe_scalar(value, default=0.0):
            if isinstance(value, (list, tuple, np.ndarray)):
                if hasattr(value, '__len__') and len(value) > 0:
                    return float(value[0])
                else:
                    return default
            return float(value) if value is not None else default
        
        # Extract basic measurements (ensuring scalar values)
        stature_cm = safe_scalar(self.imputed_data[1])  # Already in cm
        weight_kg = safe_scalar(self.imputed_data[0])   # Already in kg
        chest_cm = safe_scalar(self.imputed_data[3])    # Already in cm  
        waist_cm = safe_scalar(self.imputed_data[4])    # Already in cm
        hip_cm = safe_scalar(self.imputed_data[5])      # Already in cm
        thigh_cm = safe_scalar(self.imputed_data[7])    # Already in cm
        
        # Create feature mapping to match CAESAR trained models
        # The trained models expect features in mm, so convert cm to mm where needed
        feature_mapping = {
            'Stature': stature_cm * 10 if stature_cm > 0 else 1700,  # Convert to mm
            'Weight': weight_kg if weight_kg > 0 else 70,            # Keep in kg
            'Chest Circumference': chest_cm * 10 if chest_cm > 0 else 950,  # Convert to mm
            'Waist Circumference, Pref': waist_cm * 10 if waist_cm > 0 else 800,  # Convert to mm
            'Hip Circumference, Maximum': hip_cm * 10 if hip_cm > 0 else 950,     # Convert to mm
            'Thigh Circumference': thigh_cm * 10 if thigh_cm > 0 else 550,        # Convert to mm
        }
        
        # Calculate BMI
        if stature_cm > 0 and weight_kg > 0:
            feature_mapping['BMI'] = weight_kg / ((stature_cm / 100) ** 2)
        else:
            feature_mapping['BMI'] = 22.0  # Default BMI
        
        # Calculate ratios
        waist_mm = feature_mapping['Waist Circumference, Pref']
        hip_mm = feature_mapping['Hip Circumference, Maximum']
        chest_mm = feature_mapping['Chest Circumference']
        stature_mm = feature_mapping['Stature']
        
        if waist_mm > 0 and hip_mm > 0:
            feature_mapping['Waist/Hip'] = waist_mm / hip_mm
        else:
            feature_mapping['Waist/Hip'] = 0.8
        
        if chest_mm > 0 and waist_mm > 0:
            feature_mapping['Chest/Waist'] = chest_mm / waist_mm
        else:
            feature_mapping['Chest/Waist'] = 1.2
        
        # Estimate missing anthropometric features based on stature and build
        if stature_mm > 0:
            feature_mapping['Shoulder Breadth'] = stature_mm * 0.23  # Typical ratio
            feature_mapping['Hip Breadth, Sitting'] = hip_mm * 0.45 if hip_mm > 0 else stature_mm * 0.18
            feature_mapping['Ankle Circumference'] = stature_mm * 0.13
            feature_mapping['Crotch Height'] = stature_mm * 0.47
            feature_mapping['Knee Height'] = stature_mm * 0.28
            feature_mapping['Arm Length (Spine to Wrist)'] = stature_mm * 0.44
        else:
            # Default values if no stature available
            feature_mapping['Shoulder Breadth'] = 380.0
            feature_mapping['Hip Breadth, Sitting'] = 350.0
            feature_mapping['Ankle Circumference'] = 220.0
            feature_mapping['Crotch Height'] = 750.0
            feature_mapping['Knee Height'] = 460.0
            feature_mapping['Arm Length (Spine to Wrist)'] = 730.0
        
        if chest_mm > 0:
            feature_mapping['Armscye Circumference'] = chest_mm * 0.42
        else:
            feature_mapping['Armscye Circumference'] = 350.0
        
        # Add height/stature ratio for models that use it
        if stature_mm > 0:
            feature_mapping['SH/S'] = feature_mapping['Shoulder Breadth'] / stature_mm
        else:
            feature_mapping['SH/S'] = 0.22
        
        # Ensure all values are scalar floats
        for key in feature_mapping:
            feature_mapping[key] = float(feature_mapping[key])
        
        return feature_mapping
    
    def _predict_single_measurement(self, measurement, input_features):
        """Predict a single somatotype measurement with realistic confidence"""
        if measurement not in self.somatotype_models:
            return {'value': None, 'confidence': 0.0}
        
        model_data = self.somatotype_models[measurement]
        
        if model_data.get('model') == 'fallback':
            # Use fallback prediction with low confidence
            value = self._fallback_prediction(measurement, input_features)
            confidence = 0.2
        else:
            # Use trained model
            predictors = model_data['predictors']
            
            # Extract features for this model - ensure all values are scalars
            X_values = []
            for p in predictors:
                feature_val = input_features.get(p, 0)
                # Ensure it's a scalar value
                if isinstance(feature_val, (list, tuple, np.ndarray)):
                    if hasattr(feature_val, '__len__') and len(feature_val) > 0:
                        feature_val = float(feature_val[0])
                    else:
                        feature_val = 0.0
                else:
                    feature_val = float(feature_val)
                X_values.append(feature_val)
            
            # Create properly shaped 2D array
            X = np.array(X_values).reshape(1, -1)
            
            try:
                # Scale features
                X_scaled = model_data['scaler'].transform(X)
                
                # Make prediction
                prediction_result = model_data['model'].predict(X_scaled)
                value = float(prediction_result[0]) if hasattr(prediction_result, '__len__') else float(prediction_result)
                
                # Get realistic confidence based on actual model performance
                confidence = self._get_model_confidence(measurement)
                
            except Exception as e:
                print(f"Model prediction failed for {measurement}: {e}")
                # Fall back to heuristic prediction
                value = self._fallback_prediction(measurement, input_features)
                confidence = 0.2
        
        # Apply bounds checking
        value = self._apply_measurement_bounds(measurement, value)
        
        return {
            'value': round(float(value), 2),
            'confidence': round(float(confidence), 3)
        }
    
    def _fallback_prediction(self, measurement, input_features):
        """Generate fallback predictions using simple heuristics"""
        stature = input_features.get('Stature', 1650) / 10  # Convert to cm
        weight = input_features.get('Weight', 70)
        bmi = weight / ((stature / 100) ** 2)
        
        if 'triceps_skinfold' in measurement:
            return max(5, 8 + (bmi - 22) * 1.2) if self.gender == 'male' else max(8, 15 + (bmi - 22) * 1.5)
        elif 'subscapular_skinfold' in measurement:
            return max(6, 10 + (bmi - 22) * 1.3) if self.gender == 'male' else max(10, 18 + (bmi - 22) * 1.4)
        elif 'suprailiac_skinfold' in measurement:
            return max(8, 12 + (bmi - 22) * 1.5) if self.gender == 'male' else max(12, 22 + (bmi - 22) * 1.6)
        elif 'calf_skinfold' in measurement:
            return max(4, 6 + (bmi - 22) * 0.8) if self.gender == 'male' else max(8, 12 + (bmi - 22) * 1.0)
        elif 'humerus_biepicondylar_breadth' in measurement:
            return 6.8 + (stature - 170) * 0.02 if self.gender == 'male' else 6.0 + (stature - 160) * 0.015
        elif 'femur_biepicondylar_breadth' in measurement:
            return 9.8 + (stature - 170) * 0.025 if self.gender == 'male' else 8.8 + (stature - 160) * 0.02
        elif 'arm_circumference_flexed' in measurement:
            return 31 + (weight - 75) * 0.3 if self.gender == 'male' else 26 + (weight - 60) * 0.25
        elif 'calf_circumference' in measurement:
            return 36 + (weight - 75) * 0.2 if self.gender == 'male' else 34 + (weight - 60) * 0.15
        else:
            return 10.0
    
    def _apply_measurement_bounds(self, measurement, value):
        """Apply physiological bounds to measurements"""
        bounds = {
            'triceps_skinfold_mm': (2, 45),
            'subscapular_skinfold_mm': (3, 55),
            'suprailiac_skinfold_mm': (4, 60),
            'calf_skinfold_mm': (2, 40),
            'humerus_biepicondylar_breadth_cm': (4.5, 8.5),
            'femur_biepicondylar_breadth_cm': (7.0, 12.0),
            'arm_circumference_flexed_cm': (18, 45),
            'calf_circumference_cm': (25, 50)
        }
        
        min_val, max_val = bounds.get(measurement, (0, 100))
        return max(min_val, min(max_val, value))
    
    def _validate_measurements(self, measurements):
        """Validate consistency of somatotype measurements"""
        validation = {'status': 'success', 'warnings': [], 'errors': []}
        
        # Check for None values
        none_measurements = [k for k, v in measurements.items() if v is None]
        if none_measurements:
            validation['warnings'].append(f"Failed to predict: {', '.join(none_measurements)}")
        
        # Check physiological relationships
        if measurements.get('triceps_skinfold_mm') and measurements.get('subscapular_skinfold_mm'):
            if measurements['subscapular_skinfold_mm'] > measurements['triceps_skinfold_mm'] * 3:
                validation['warnings'].append("Subscapular skinfold unusually high relative to triceps")
        
        if measurements.get('arm_circumference_flexed_cm') and measurements.get('calf_circumference_cm'):
            if measurements['arm_circumference_flexed_cm'] < measurements['calf_circumference_cm'] * 0.6:
                validation['warnings'].append("Arm circumference seems low relative to calf")
        
        return validation
    
    def save_somatotype_results(self, results, output_name="somatotype_measurements"):
        """Save somatotype measurements with real performance metrics to output folder"""
        try:
            output_file = os.path.join(OUTPUT_FILES_DIR, f"{output_name}_{self.gender}.csv")
            
            # Prepare data for CSV
            data = []
            
            # Base measurements
            base_data = {
                'measurement_type': 'base',
                'weight_kg': float(self.imputed_data[0]),
                'stature_cm': float(self.imputed_data[1]),
                'chest_girth_cm': float(self.imputed_data[3]),
                'waist_girth_cm': float(self.imputed_data[4]),
                'hips_buttock_girth_cm': float(self.imputed_data[5]),
                'gender': self.gender
            }
            data.append(base_data)
            
            # Somatotype measurements
            if 'somatotype_measurements' in results:
                somatotype_data = {'measurement_type': 'somatotype', 'gender': self.gender}
                
                for measurement, value in results['somatotype_measurements'].items():
                    if value is not None:
                        somatotype_data[measurement] = value
                        # Add real confidence score based on model performance
                        confidence_key = f"{measurement}_confidence"
                        somatotype_data[confidence_key] = results.get('confidence_scores', {}).get(measurement, 0)
                
                data.append(somatotype_data)
            
            # Save to CSV
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
            print(f"Somatotype results saved to: {output_file}")
            
            # Also save a detailed summary text file with model performance
            summary_file = os.path.join(OUTPUT_FILES_DIR, f"{output_name}_{self.gender}_summary.txt")
            with open(summary_file, 'w') as f:
                f.write("=== SOMATOTYPE MEASUREMENT RESULTS ===\n\n")
                f.write(f"Gender: {self.gender}\n")
                f.write(f"Processing Date: {pd.Timestamp.now()}\n\n")
                
                f.write("BASE MEASUREMENTS:\n")
                f.write(f"  Weight: {float(self.imputed_data[0]):.1f} kg\n")
                f.write(f"  Stature: {float(self.imputed_data[1]):.1f} cm\n")
                f.write(f"  Chest Girth: {float(self.imputed_data[3]):.1f} cm\n")
                f.write(f"  Waist Girth: {float(self.imputed_data[4]):.1f} cm\n")
                f.write(f"  Hip Girth: {float(self.imputed_data[5]):.1f} cm\n\n")
                
                if 'somatotype_measurements' in results:
                    f.write("SOMATOTYPE MEASUREMENTS:\n")
                    f.write("\nSkinfold Measurements:\n")
                    for measurement in ['triceps_skinfold_mm', 'subscapular_skinfold_mm', 
                                       'suprailiac_skinfold_mm', 'calf_skinfold_mm']:
                        value = results['somatotype_measurements'].get(measurement)
                        confidence = results.get('confidence_scores', {}).get(measurement, 0)
                        if value is not None:
                            f.write(f"  {measurement.replace('_', ' ').title()}: {value:.1f} mm (confidence: {confidence:.2f})\n")
                    
                    f.write("\nBone Breadth Measurements:\n")
                    for measurement in ['humerus_biepicondylar_breadth_cm', 'femur_biepicondylar_breadth_cm']:
                        value = results['somatotype_measurements'].get(measurement)
                        confidence = results.get('confidence_scores', {}).get(measurement, 0)
                        if value is not None:
                            f.write(f"  {measurement.replace('_', ' ').title()}: {value:.1f} cm (confidence: {confidence:.2f})\n")
                    
                    f.write("\nSpecialized Circumferences:\n")
                    for measurement in ['arm_circumference_flexed_cm', 'calf_circumference_cm']:
                        value = results['somatotype_measurements'].get(measurement)
                        confidence = results.get('confidence_scores', {}).get(measurement, 0)
                        if value is not None:
                            f.write(f"  {measurement.replace('_', ' ').title()}: {value:.1f} cm (confidence: {confidence:.2f})\n")
                
                # Add model performance details
                f.write("\n" + "="*50 + "\n")
                f.write("MODEL PERFORMANCE METRICS\n")
                f.write("="*50 + "\n")
                
                if hasattr(self, 'training_report') and 'model_performance' in self.training_report:
                    f.write("Based on CAESAR dataset training:\n\n")
                    for measurement, perf in self.training_report['model_performance'].items():
                        if measurement in results.get('somatotype_measurements', {}):
                            f.write(f"{measurement.replace('_', ' ').title()}:\n")
                            f.write(f"  Model Type: {perf.get('model_type', 'Unknown')}\n")
                            f.write(f"  R² Score: {perf.get('test_r2', 0):.3f}\n")
                            f.write(f"  Mean Absolute Error: {perf.get('test_mae', 0):.2f}\n")
                            f.write(f"  Training Samples: {perf.get('n_samples', 0)}\n")
                            f.write(f"  Features Used: {perf.get('n_features', 0)}\n\n")
                else:
                    f.write("Model performance data not available.\n")
                
                # Validation status
                if 'validation_status' in results:
                    f.write(f"VALIDATION STATUS: {results['validation_status']['status']}\n")
                    if results['validation_status']['warnings']:
                        f.write("Warnings:\n")
                        for warning in results['validation_status']['warnings']:
                            f.write(f"  - {warning}\n")
                    if results['validation_status']['errors']:
                        f.write("Errors:\n")
                        for error in results['validation_status']['errors']:
                            f.write(f"  - {error}\n")
            
            print(f"Detailed summary saved to: {summary_file}")
            
        except Exception as e:
            print(f"Error saving somatotype results: {e}")
    
    def get_model_performance_summary(self):
        """Get a summary of model performance metrics"""
        if not hasattr(self, 'training_report') or 'model_performance' not in self.training_report:
            return "Model performance data not available"
        
        summary = []
        summary.append("SOMATOTYPE MODEL PERFORMANCE SUMMARY")
        summary.append("=" * 50)
        summary.append(f"Gender: {self.gender}")
        summary.append(f"Training Date: {self.training_report.get('training_date', 'Unknown')}")
        summary.append(f"Dataset Size: {self.training_report.get('dataset_info', {}).get('total_subjects', 'Unknown')} subjects")
        summary.append("")
        
        for measurement, perf in self.training_report['model_performance'].items():
            summary.append(f"{measurement.replace('_', ' ').title()}:")
            summary.append(f"  Model: {perf.get('model_type', 'Unknown')}")
            summary.append(f"  R² Score: {perf.get('test_r2', 0):.3f}")
            summary.append(f"  MAE: {perf.get('test_mae', 0):.2f}")
            summary.append(f"  Samples: {perf.get('n_samples', 0)}")
            summary.append("")
        
        return "\n".join(summary)


def create_enhanced_avatar_from_measurements(measurements, gender="female", enable_somatotype=True):
    """
    Convenience function to create an enhanced avatar from measurements.
    
    Args:
        measurements (list or array): Basic measurements
        gender (str): "female" or "male"
        enable_somatotype (bool): Enable somatotype predictions
    
    Returns:
        AvatarSomatotype: Enhanced avatar instance
    """
    # Convert to numpy array if needed
    input_data = np.array(measurements) if not isinstance(measurements, np.ndarray) else measurements
    
    # Pad with zeros if needed to match M_NUM
    if len(input_data) < M_NUM:
        padded_data = np.zeros(M_NUM)
        padded_data[:len(input_data)] = input_data
        input_data = padded_data
    
    return AvatarSomatotype(input_data, gender, enable_somatotype)


def demo_somatotype_system():
    """Demonstration of the complete somatotype system"""
    print("=== CAESAR-Based Somatotype System Demo ===\n")
    
    # Example measurements (weight, stature, neck, chest, waist, hips, shoulder, thigh, etc.)
    example_measurements = [65.0, 165.0, 32.0, 88.0, 72.0, 96.0, 0, 56.0, 0, 36.0, 22.0, 
                           26.0, 16.0, 0, 0, 0, 0, 0, 0, 0, 55.0]
    
    print("Input measurements:")
    measurement_names = ['weight_kg', 'stature_cm', 'neck_base_girth', 'chest_girth', 'waist_girth', 
                        'hips_buttock_girth', 'shoulder_girth', 'thigh_girth']
    for i, name in enumerate(measurement_names):
        if i < len(example_measurements) and example_measurements[i] != 0:
            print(f"  {name}: {example_measurements[i]}")
    
    print("\nCreating enhanced avatar with somatotype capabilities...")
    
    # Create enhanced avatar
    avatar = create_enhanced_avatar_from_measurements(example_measurements, gender="female", enable_somatotype=True)
    
    # Get complete predictions
    print("\nRunning complete prediction (base + somatotype)...")
    results = avatar.predict_complete(include_somatotype=True)
    
    # Display results
    print("\n=== RESULTS ===")
    print(f"Base measurements completed: ✓")
    
    if results['somatotype_measurements']:
        print("\nSomatotype Measurements:")
        for measurement, value in results['somatotype_measurements'].items():
            if value is not None:
                confidence = results['confidence_scores'].get(measurement, 0)
                unit = 'mm' if 'skinfold' in measurement else 'cm'
                print(f"  {measurement.replace('_', ' ').title()}: {value:.1f} {unit} (confidence: {confidence:.2f})")
    
    print(f"\nValidation: {results['validation_status']['status']}")
    if results['validation_status']['warnings']:
        print("Warnings:", results['validation_status']['warnings'])
    
    # Save results
    avatar.save_somatotype_results(results, "demo_somatotype_output")
    
    # Create 3D avatar
    print("\nGenerating 3D avatar...")
    avatar.create_obj_file(ava_name=f'avatar_somatotype_demo_{avatar.gender}')
    
    print("Demo completed successfully!")
    return results


if __name__ == "__main__":
    # Run demonstration
    demo_results = demo_somatotype_system()
