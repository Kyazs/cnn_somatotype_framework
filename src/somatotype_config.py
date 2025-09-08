"""
Configuration file for the CAESAR-based somatotype measurement system.

This file contains all configurable parameters for the somatotype system,
allowing easy customization without modifying the core code.
"""

# Somatotype measurement definitions
SOMATOTYPE_MEASUREMENTS = {
    'triceps_skinfold_mm': {
        'name': 'Triceps Skinfold',
        'unit': 'mm',
        'category': 'skinfold',
        'description': 'Upper arm posterior fat thickness',
        'bounds': (2, 45),
        'typical_range': {'female': (15, 30), 'male': (8, 20)}
    },
    'subscapular_skinfold_mm': {
        'name': 'Subscapular Skinfold',
        'unit': 'mm', 
        'category': 'skinfold',
        'description': 'Shoulder blade region fat thickness',
        'bounds': (3, 55),
        'typical_range': {'female': (12, 28), 'male': (10, 25)}
    },
    'suprailiac_skinfold_mm': {
        'name': 'Suprailiac Skinfold',
        'unit': 'mm',
        'category': 'skinfold', 
        'description': 'Hip bone region fat thickness',
        'bounds': (4, 60),
        'typical_range': {'female': (15, 35), 'male': (10, 28)}
    },
    'calf_skinfold_mm': {
        'name': 'Calf Skinfold',
        'unit': 'mm',
        'category': 'skinfold',
        'description': 'Lower leg posterior fat thickness',
        'bounds': (2, 40),
        'typical_range': {'female': (12, 25), 'male': (6, 16)}
    },
    'humerus_biepicondylar_breadth_cm': {
        'name': 'Humerus Biepicondylar Breadth',
        'unit': 'cm',
        'category': 'bone_breadth',
        'description': 'Elbow width (skeletal frame)',
        'bounds': (4.5, 8.5),
        'typical_range': {'female': (5.8, 6.6), 'male': (6.5, 7.5)}
    },
    'femur_biepicondylar_breadth_cm': {
        'name': 'Femur Biepicondylar Breadth', 
        'unit': 'cm',
        'category': 'bone_breadth',
        'description': 'Knee width (skeletal frame)',
        'bounds': (7.0, 12.0),
        'typical_range': {'female': (8.5, 9.7), 'male': (9.5, 10.9)}
    },
    'arm_circumference_flexed_cm': {
        'name': 'Arm Circumference, Flexed and Tensed',
        'unit': 'cm',
        'category': 'circumference',
        'description': 'Maximum contracted arm circumference',
        'bounds': (18, 45),
        'typical_range': {'female': (25, 32), 'male': (30, 38)}
    },
    'calf_circumference_cm': {
        'name': 'Calf Circumference',
        'unit': 'cm', 
        'category': 'circumference',
        'description': 'Maximum lower leg circumference',
        'bounds': (25, 50),
        'typical_range': {'female': (32, 40), 'male': (35, 42)}
    }
}

# Predictor sets for each measurement (optimized for CAESAR data)
PREDICTOR_SETS = {
    'triceps_skinfold_mm': {
        'primary': ['Stature', 'Weight', 'BMI', 'Armscye Circumference', 'Chest Circumference'],
        'secondary': ['Shoulder Breadth', 'Waist Circumference, Pref', 'Chest/Waist'],
        'fallback': ['Stature', 'Weight', 'BMI']
    },
    'subscapular_skinfold_mm': {
        'primary': ['Stature', 'Weight', 'BMI', 'Chest Circumference', 'Waist Circumference, Pref'],
        'secondary': ['Shoulder Breadth', 'Hip Circumference, Maximum', 'Chest/Waist'],
        'fallback': ['Stature', 'Weight', 'BMI']
    },
    'suprailiac_skinfold_mm': {
        'primary': ['Weight', 'BMI', 'Waist Circumference, Pref', 'Hip Circumference, Maximum', 'Chest/Waist'],
        'secondary': ['Stature', 'Chest Circumference', 'Waist/Hip'],
        'fallback': ['Weight', 'BMI', 'Waist Circumference, Pref']
    },
    'calf_skinfold_mm': {
        'primary': ['Weight', 'BMI', 'Thigh Circumference', 'Ankle Circumference'],
        'secondary': ['Stature', 'Triceps Skinfold', 'Hip Circumference, Maximum'],
        'fallback': ['Weight', 'BMI', 'Stature']
    },
    'humerus_biepicondylar_breadth_cm': {
        'primary': ['Stature', 'Shoulder Breadth', 'Armscye Circumference', 'Arm Length (Spine to Wrist)'],
        'secondary': ['Chest Circumference', 'Weight'],
        'fallback': ['Stature', 'Shoulder Breadth']
    },
    'femur_biepicondylar_breadth_cm': {
        'primary': ['Stature', 'Hip Breadth, Sitting', 'Crotch Height', 'Knee Height'],
        'secondary': ['Hip Circumference, Maximum', 'Weight'],
        'fallback': ['Stature', 'Hip Breadth, Sitting']
    },
    'arm_circumference_flexed_cm': {
        'primary': ['Armscye Circumference', 'Chest Circumference', 'Weight', 'BMI'],
        'secondary': ['Shoulder Breadth', 'Triceps Skinfold', 'Stature'],
        'fallback': ['Armscye Circumference', 'Weight']
    },
    'calf_circumference_cm': {
        'primary': ['Thigh Circumference', 'Ankle Circumference', 'Stature', 'Weight'],
        'secondary': ['Calf Skinfold', 'Hip Circumference, Maximum'],
        'fallback': ['Stature', 'Weight']
    }
}

# Model configuration for each measurement type
MODEL_CONFIGS = {
    'skinfold': {
        'model_type': 'RandomForest',
        'parameters': {
            'n_estimators': 100,
            'random_state': 42,
            'max_depth': 10,
            'min_samples_split': 5
        },
        'expected_accuracy': {'mae': 3.5, 'r2': 0.75}
    },
    'bone_breadth': {
        'model_type': 'Ridge',
        'parameters': {
            'alpha': 1.0,
            'random_state': 42
        },
        'expected_accuracy': {'mae': 0.3, 'r2': 0.85}
    },
    'circumference': {
        'model_type': 'GradientBoosting',
        'parameters': {
            'n_estimators': 100,
            'random_state': 42,
            'learning_rate': 0.1,
            'max_depth': 6
        },
        'expected_accuracy': {'mae': 1.5, 'r2': 0.80}
    }
}

# System configuration
SYSTEM_CONFIG = {
    'enable_caching': True,
    'cache_models': True,
    'default_confidence_threshold': 0.75,
    'validation_mode': 'strict',  # 'strict', 'moderate', 'permissive'
    'fallback_enabled': True,
    'output_precision': 1,  # decimal places
    'processing_timeout': 120,  # seconds
}

# Feature engineering configuration
FEATURE_ENGINEERING = {
    'enable_derived_features': True,
    'anthropometric_ratios': {
        'bmi': True,
        'waist_hip_ratio': True,
        'chest_waist_ratio': True,
        'shoulder_frame_ratio': True,
        'hip_frame_ratio': True
    },
    'scaling_factors': {
        'height_based_scaling': True,
        'weight_based_scaling': True,
        'gender_specific_scaling': True
    }
}

# Validation rules
VALIDATION_RULES = {
    'physiological_bounds_check': True,
    'inter_measurement_consistency': True,
    'population_distribution_check': True,
    'outlier_detection': True,
    'correlation_validation': {
        'triceps_subscapular_correlation': {'min': 0.6, 'max': 0.95},
        'skinfold_bmi_correlation': {'min': 0.4, 'max': 0.85},
        'bone_breadth_height_correlation': {'min': 0.5, 'max': 0.90}
    }
}

# Output configuration
OUTPUT_CONFIG = {
    'save_detailed_csv': True,
    'save_summary_text': True,
    'save_confidence_scores': True,
    'save_validation_report': True,
    'include_intermediate_results': False,
    'output_formats': ['csv', 'json', 'txt'],
    'filename_template': 'somatotype_measurements_{gender}_{timestamp}'
}

# Gender-specific defaults for fallback predictions
GENDER_DEFAULTS = {
    'female': {
        'triceps_skinfold_mm': {'mean': 23, 'std': 8},
        'subscapular_skinfold_mm': {'mean': 20, 'std': 9},
        'suprailiac_skinfold_mm': {'mean': 25, 'std': 10},
        'calf_skinfold_mm': {'mean': 18, 'std': 7},
        'humerus_biepicondylar_breadth_cm': {'mean': 6.2, 'std': 0.5},
        'femur_biepicondylar_breadth_cm': {'mean': 9.1, 'std': 0.7},
        'arm_circumference_flexed_cm': {'mean': 28, 'std': 3},
        'calf_circumference_cm': {'mean': 36, 'std': 3}
    },
    'male': {
        'triceps_skinfold_mm': {'mean': 12, 'std': 6},
        'subscapular_skinfold_mm': {'mean': 15, 'std': 7},
        'suprailiac_skinfold_mm': {'mean': 18, 'std': 8},
        'calf_skinfold_mm': {'mean': 10, 'std': 5},
        'humerus_biepicondylar_breadth_cm': {'mean': 7.1, 'std': 0.6},
        'femur_biepicondylar_breadth_cm': {'mean': 10.2, 'std': 0.8},
        'arm_circumference_flexed_cm': {'mean': 33, 'std': 4},
        'calf_circumference_cm': {'mean': 38, 'std': 3}
    }
}

# CAESAR data configuration
CAESAR_CONFIG = {
    'data_files': {
        'female': 'measurements_CAESAR_female_somatotype.csv',
        'male': 'measurements_CAESAR_male_somatotype.csv',
        'enhanced_female': 'measurements_CAESAR_female_enhanced.csv',
        'enhanced_male': 'measurements_CAESAR_male_enhanced.csv'
    },
    'required_columns': ['Stature', 'Weight', 'BMI'],
    'preferred_columns': [
        'Shoulder Breadth', 'Armscye Circumference', 'Hip Breadth, Sitting',
        'Hip Circumference, Maximum', 'Waist Circumference, Pref', 
        'Chest Circumference', 'Thigh Circumference', 'Ankle Circumference'
    ],
    'data_validation': {
        'remove_outliers': True,
        'outlier_threshold': 3.0,  # standard deviations
        'min_valid_records': 100,
        'required_completeness': 0.8  # 80% of data must be complete
    }
}

# Error handling configuration
ERROR_HANDLING = {
    'continue_on_model_failure': True,
    'use_fallback_on_error': True,
    'log_errors': True,
    'raise_critical_errors': True,
    'retry_attempts': 2,
    'timeout_handling': 'graceful'  # 'graceful' or 'strict'
}

# Performance configuration
PERFORMANCE_CONFIG = {
    'parallel_processing': False,  # Enable when using multiple subjects
    'batch_size': 32,
    'memory_optimization': True,
    'model_compression': False,
    'prediction_caching': True
}
