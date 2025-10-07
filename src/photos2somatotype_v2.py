#!/usr/bin/env python3
"""
Hybrid CNN-XGBoost Anthropometric Measurement System (Version 2)
=================================================================

This script combines two machine learning models for comprehensive body measurement extraction:

1. CNN Extractor Model: Extracts basic anthropometric measurements from silhouette images
2. XGBoost Predictor Model: Refines measurements and predicts specialized somatotype measurements

Processing Pipeline:
--------------------
INPUT:
  - Front and side view photos
  - Gender, weight, age, and optionally stature (from CSV)

STEP 1 - Image Processing:
  - Extract silhouettes using DeepLabV3 segmentation
  - Resize to 224x224 for CNN input

STEP 2 - CNN Extraction:
  Input: [gender, stature_cm (optional)] + [front_image, side_image]
  Output: 10 basic measurements including stature, circumferences, breadths

STEP 3 - BMI Calculation:
  BMI = weight_kg / (stature_cm/100)²

STEP 4 - XGBoost Prediction:
  Input: CNN measurements + BMI + Age + Gender
  Output: 8 somatotype measurements (4 skinfolds + 2 breadths + 2 circumferences)

FINAL OUTPUT:
  - All CNN measurements
  - All XGBoost predictions
  - Refined measurements (averaged for overlapping items)
  - Comprehensive CSV report

Author: Your Name
Date: October 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import glob
from PIL import Image
import tensorflow as tf
import torch
from torchvision import transforms
from sklearn.preprocessing import StandardScaler

# GPU Configuration
gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("Running on CPU")
else:
    print(f"GPU available: {gpus}")

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

# Import utilities
from utils import (
    INPUT_FILES_DIR,
    OUTPUT_FILES_DIR,
    MODEL_FILES_DIR,
    GENDER_DICT,
    IMG_SIZE_4NN,
    UK_MEAS,
)

# Import the XGBoost predictor
from measurement_prediction.train_predictor_model import BodyMeasurementPredictor

# Constants
IMG_RESIZE = 448
SEGMENTATION_MODEL_PATH = os.path.join(MODEL_FILES_DIR, "deeplabv3_resnet101")


class HybridMeasurementSystem:
    """
    Hybrid system combining CNN extractor and XGBoost predictor for
    comprehensive anthropometric measurement extraction.
    """

    def __init__(self):
        self.cnn_model = None
        self.xgboost_predictor = None
        self.scaler = None
        self.segmentation_model = None
        self.input_info = {}
        self.cnn_measurements = {}
        self.xgboost_measurements = {}
        self.final_measurements = {}

    def load_input_info(self, csv_path=None):
        """
        Load input information from CSV file.

        Expected format: gender,weight_kg,age,stature_cm
        Note: stature_cm is optional (leave empty if CNN should predict it)
        """
        if csv_path is None:
            csv_path = os.path.join(INPUT_FILES_DIR, "input_info_extractor.csv")

        print(f"\n{'='*70}")
        print("Loading Input Information")
        print(f"{'='*70}")
        print(f"Reading: {csv_path}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Input file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Extract values
        self.input_info["gender"] = df["gender"].iloc[0].strip().lower()
        self.input_info["weight_kg"] = float(df["weight_kg"].iloc[0])
        self.input_info["age"] = int(df["age"].iloc[0])

        # Stature is optional (CNN will predict if not provided)
        stature_value = df["stature_cm"].iloc[0]
        if pd.isna(stature_value) or str(stature_value).strip() == "":
            self.input_info["stature_cm"] = None
            print("  Stature: Will be predicted by CNN")
        else:
            self.input_info["stature_cm"] = float(stature_value)
            print(f"  Stature: {self.input_info['stature_cm']} cm (user provided)")

        # Convert gender to numeric
        self.input_info["gender_numeric"] = GENDER_DICT[self.input_info["gender"]]
        self.input_info["gender_male"] = 1 if self.input_info["gender"] == "male" else 0

        print(f"\nInput Information Loaded:")
        print(f"  Gender: {self.input_info['gender']}")
        print(f"  Weight: {self.input_info['weight_kg']} kg")
        print(f"  Age: {self.input_info['age']} years")

        return self.input_info

    def load_cnn_model(self, model_name=None):
        """Load the trained CNN extractor model."""
        print(f"\n{'='*70}")
        print("Loading CNN Extractor Model")
        print(f"{'='*70}")

        if model_name is None:
            # Find the latest extractor model
            model_pattern = os.path.join(MODEL_FILES_DIR, "extractor_*.keras")
            model_files = glob.glob(model_pattern)

            if not model_files:
                raise FileNotFoundError(
                    f"No extractor models found in {MODEL_FILES_DIR}"
                )

            model_path = max(model_files, key=os.path.getmtime)
        else:
            model_path = os.path.join(MODEL_FILES_DIR, model_name)

        print(f"Loading: {os.path.basename(model_path)}")

        try:
            self.cnn_model = tf.keras.models.load_model(model_path, compile=False)
            print(f"✅ CNN model loaded successfully")
            print(f"   Input shape: {self.cnn_model.input_shape}")
            print(f"   Output shape: {self.cnn_model.output_shape}")
        except Exception as e:
            raise RuntimeError(f"Failed to load CNN model: {e}")

        # Load the scaler
        scaler_pattern = os.path.join(MODEL_FILES_DIR, "scaler*_extractor.pkl")
        scaler_files = glob.glob(scaler_pattern)

        if scaler_files:
            scaler_path = scaler_files[0]
            try:
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                print(f"✅ Scaler loaded: {os.path.basename(scaler_path)}")
            except (pickle.UnpicklingError, AttributeError, ImportError) as e:
                print(f"⚠️  Failed to load scaler ({e}), will use raw values")
                self.scaler = None
        else:
            print("⚠️  No scaler found, will use raw values")
            self.scaler = None

        return self.cnn_model

    def load_xgboost_model(self):
        """Load the trained XGBoost predictor model."""
        print(f"\n{'='*70}")
        print("Loading XGBoost Predictor Model")
        print(f"{'='*70}")

        # Find the latest XGBoost model
        model_pattern = os.path.join(MODEL_FILES_DIR, "xgboost_body_predictor_*.pkl")
        model_files = glob.glob(model_pattern)

        if not model_files:
            raise FileNotFoundError(f"No XGBoost models found in {MODEL_FILES_DIR}")

        model_path = max(model_files, key=os.path.getmtime)
        print(f"Loading: {os.path.basename(model_path)}")

        self.xgboost_predictor = BodyMeasurementPredictor()
        self.xgboost_predictor.load_model(model_path)

        print(f"✅ XGBoost model loaded successfully")
        print(
            f"   Proxy measurements: {len(self.xgboost_predictor.proxy_measurements)}"
        )
        print(
            f"   Target measurements: {len(self.xgboost_predictor.target_measurements)}"
        )

        return self.xgboost_predictor

    def load_segmentation_model(self):
        """Load DeepLabV3 segmentation model for silhouette extraction."""
        print(f"\n{'='*70}")
        print("Loading Segmentation Model (DeepLabV3)")
        print(f"{'='*70}")

        try:
            self.segmentation_model = torch.hub.load(
                "pytorch/vision:v0.10.0", "deeplabv3_resnet101", pretrained=True
            )
            self.segmentation_model.eval()
            print("✅ Segmentation model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load segmentation model: {e}")

        return self.segmentation_model

    def load_and_process_images(self):
        """Load front and side images and extract silhouettes."""
        print(f"\n{'='*70}")
        print("Loading and Processing Images")
        print(f"{'='*70}")

        gender = self.input_info["gender"]

        # Load images
        front_path = os.path.join(INPUT_FILES_DIR, f"input_front.png")
        side_path = os.path.join(INPUT_FILES_DIR, f"input_side.png")

        if not os.path.exists(front_path):
            raise FileNotFoundError(f"Front image not found: {front_path}")
        if not os.path.exists(side_path):
            raise FileNotFoundError(f"Side image not found: {side_path}")

        print(f"Loading front image: {front_path}")
        print(f"Loading side image: {side_path}")

        front_img = Image.open(front_path).convert("RGB")
        side_img = Image.open(side_path).convert("RGB")

        # Extract silhouettes
        print("\nExtracting silhouettes...")
        front_sil = self._extract_silhouette(front_img)
        side_sil = self._extract_silhouette(side_img)

        # Prepare for CNN (224x224, normalized)
        front_cnn = self._prepare_for_cnn(front_sil)
        side_cnn = self._prepare_for_cnn(side_sil)

        # Save silhouettes for visualization
        front_sil.save(os.path.join(OUTPUT_FILES_DIR, f"sil_front_{gender}.png"))
        side_sil.save(os.path.join(OUTPUT_FILES_DIR, f"sil_side_{gender}.png"))

        print(f"✅ Images processed successfully")
        print(f"   Front CNN input: {front_cnn.shape}")
        print(f"   Side CNN input: {side_cnn.shape}")

        return front_cnn, side_cnn

    def _extract_silhouette(self, img):
        """Extract silhouette from image using DeepLabV3."""
        # Prepare image for segmentation
        preprocess = transforms.Compose(
            [
                transforms.Resize((IMG_RESIZE, IMG_RESIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)

        # Run segmentation
        with torch.no_grad():
            output = self.segmentation_model(input_batch)["out"][0]

        # Get person mask (class 15 in COCO)
        output_predictions = output.argmax(0).byte().cpu().numpy()
        person_mask = (output_predictions == 15).astype(np.uint8) * 255

        # Convert to PIL Image
        silhouette = Image.fromarray(person_mask, mode="L")

        return silhouette

    def _prepare_for_cnn(self, silhouette):
        """Prepare silhouette for CNN input (224x224, normalized)."""
        # Resize to CNN input size
        sil_resized = silhouette.resize((IMG_SIZE_4NN, IMG_SIZE_4NN), Image.LANCZOS)

        # Convert to numpy array
        sil_array = np.array(sil_resized, dtype=np.float32)

        # Normalize to [0, 1]
        sil_array = sil_array / 255.0

        # Add channel dimension
        sil_array = np.expand_dims(sil_array, axis=-1)

        return sil_array

    def run_cnn_extraction(self, front_img, side_img):
        """Run CNN model to extract measurements."""
        print(f"\n{'='*70}")
        print("Step 1: CNN Extraction")
        print(f"{'='*70}")

        # Prepare numerical input [gender, stature_cm]
        gender_encoded = self.input_info["gender_numeric"]
        
        # Track if stature was provided by user
        user_provided_stature = self.input_info["stature_cm"] is not None

        # Use provided stature or a placeholder (CNN will predict)
        if user_provided_stature:
            stature_input = self.input_info["stature_cm"]
            print(f"Using user-provided stature: {stature_input} cm (will NOT be overridden by CNN)")
        else:
            # Use average stature as placeholder (CNN will predict)
            stature_input = 170.0
            print(f"Using placeholder stature for CNN: {stature_input} cm (CNN will predict actual value)")

        # Create numerical input array
        numerical_data = np.array([[gender_encoded, stature_input]], dtype=np.float32)

        # Scale if scaler is available
        if self.scaler is not None:
            # One-hot encode gender first
            gender_onehot = np.array([[1, 0] if gender_encoded == 0 else [0, 1]])
            stature_array = np.array([[stature_input]])

            # Scale only the continuous variable (stature)
            stature_scaled = self.scaler.transform(stature_array)

            # Combine
            numerical_input = np.hstack([gender_onehot, stature_scaled])
        else:
            # Simple encoding without scaling
            gender_onehot = np.array([[1, 0] if gender_encoded == 0 else [0, 1]])
            numerical_input = np.hstack([gender_onehot, [[stature_input]]])

        # Prepare image inputs (add batch dimension)
        front_batch = np.expand_dims(front_img, axis=0)
        side_batch = np.expand_dims(side_img, axis=0)

        print(f"\nCNN Input shapes:")
        print(f"  Numerical: {numerical_input.shape}")
        print(f"  Front image: {front_batch.shape}")
        print(f"  Side image: {side_batch.shape}")

        # Run prediction
        print("\nRunning CNN prediction...")
        predictions = self.cnn_model.predict(
            [numerical_input, front_batch, side_batch], verbose=0
        )

        # Map predictions to measurement names
        measurement_names = UK_MEAS
        predicted_values = predictions[0]

        print(f"\n✅ CNN Extraction Complete:")
        for i, (name, value) in enumerate(zip(measurement_names, predicted_values)):
            # If user provided stature, use that instead of CNN prediction
            if name == 'stature_cm' and user_provided_stature:
                self.cnn_measurements[name] = float(self.input_info["stature_cm"])
                print(f"   {name}: {self.input_info['stature_cm']:.2f} (user-provided, not CNN prediction)")
            else:
                self.cnn_measurements[name] = float(value)
                print(f"   {name}: {value:.2f}")

        return self.cnn_measurements

    def compute_bmi(self):
        """Compute BMI from weight and stature."""
        weight = self.input_info["weight_kg"]
        stature_cm = self.cnn_measurements["stature_cm"]
        stature_m = stature_cm / 100.0

        bmi = weight / (stature_m**2)

        print(f"\n{'='*70}")
        print("Step 2: BMI Calculation")
        print(f"{'='*70}")
        print(f"Weight: {weight} kg")
        print(f"Stature: {stature_cm:.2f} cm")
        print(f"BMI: {bmi:.2f}")

        return bmi

    def run_xgboost_prediction(self, bmi):
        """Run XGBoost model to predict somatotype measurements."""
        print(f"\n{'='*70}")
        print("Step 3: XGBoost Prediction")
        print(f"{'='*70}")

        # Prepare proxy measurements for XGBoost
        # Expected order from train_predictor_model.py:
        # Stature, Weight, Chest_Circumference, Hip_Circumference,
        # Waist_Circumference, Thigh_Circumference, Ankle_Circumference,
        # Shoulder_Breadth, Knee_Height, Arm_Circumference_Flexed,
        # Calf_Circumference, BMI, Age, Gender_Male

        proxy_data = {
            "Stature": self.cnn_measurements["stature_cm"],
            "Weight": self.input_info["weight_kg"],
            "Chest_Circumference": self.cnn_measurements["chest_circumference"],
            "Hip_Circumference": self.cnn_measurements["buttock_circumference"],
            "Waist_Circumference": self.cnn_measurements["waist_circumference"],
            "Thigh_Circumference": self.cnn_measurements["thigh_circumference"],
            "Ankle_Circumference": self.cnn_measurements["ankle_circumference"],
            "Shoulder_Breadth": self.cnn_measurements["biacromial_breadth"],
            "Knee_Height": self.cnn_measurements["knee_height_sitting"],
            "Arm_Circumference_Flexed": self.cnn_measurements[
                "arm_circumference_flexed"
            ],
            "Calf_Circumference": self.cnn_measurements["calf_circumference"],
            "BMI": bmi,
            "Age": self.input_info["age"],
            "Gender_Male": self.input_info["gender_male"],
        }

        print("\nProxy measurements for XGBoost:")
        for key, value in proxy_data.items():
            print(f"   {key}: {value:.2f}")

        # Create DataFrame in the exact order expected by the model
        X_input = pd.DataFrame([proxy_data])
        X_input = X_input[self.xgboost_predictor.proxy_measurements]

        print(f"\nInput DataFrame shape: {X_input.shape}")
        print(f"Expected columns: {self.xgboost_predictor.proxy_measurements}")

        # Get predictions
        print("\nRunning XGBoost prediction...")
        predictions = self.xgboost_predictor.predict(X_input)
        predictions_df = pd.DataFrame(
            predictions, columns=self.xgboost_predictor.target_measurements
        )

        print(f"\n✅ XGBoost Prediction Complete:")
        for col in predictions_df.columns:
            value = predictions_df[col].iloc[0]
            self.xgboost_measurements[col] = float(value)
            print(f"   {col}: {value:.2f}")

        return self.xgboost_measurements

    def combine_measurements(self):
        """Combine CNN and XGBoost measurements, averaging overlapping ones."""
        print(f"\n{'='*70}")
        print("Step 4: Combining Measurements")
        print(f"{'='*70}")

        # Start with all CNN measurements
        self.final_measurements = self.cnn_measurements.copy()

        # Add user inputs
        self.final_measurements["weight_kg"] = self.input_info["weight_kg"]
        self.final_measurements["age"] = self.input_info["age"]
        self.final_measurements["gender"] = self.input_info["gender"]

        # Add all XGBoost predictions
        for key, value in self.xgboost_measurements.items():
            self.final_measurements[f"xgb_{key}"] = value

        # Average overlapping measurements
        overlapping = ["Arm_Circumference_Flexed", "Calf_Circumference"]

        print("\nAveraging overlapping measurements:")
        for measurement in overlapping:
            cnn_key = measurement.lower().replace("_", "_")
            xgb_key = measurement

            if (
                cnn_key in self.cnn_measurements
                and xgb_key in self.xgboost_measurements
            ):
                cnn_value = self.cnn_measurements[cnn_key]
                xgb_value = self.xgboost_measurements[xgb_key]
                averaged = (cnn_value + xgb_value) / 2

                self.final_measurements[f"{cnn_key}_refined"] = averaged

                print(f"   {measurement}:")
                print(f"      CNN: {cnn_value:.2f}")
                print(f"      XGBoost: {xgb_value:.2f}")
                print(f"      Refined (averaged): {averaged:.2f}")

        return self.final_measurements

    def save_results(self):
        """Save all results to CSV files."""
        print(f"\n{'='*70}")
        print("Saving Results")
        print(f"{'='*70}")

        gender = self.input_info["gender"]

        # Save CNN measurements
        cnn_df = pd.DataFrame([self.cnn_measurements])
        cnn_path = os.path.join(OUTPUT_FILES_DIR, f"cnn_measurements_{gender}.csv")
        cnn_df.to_csv(cnn_path, index=False)
        print(f"✅ CNN measurements saved: {cnn_path}")

        # Save XGBoost measurements
        xgb_df = pd.DataFrame([self.xgboost_measurements])
        xgb_path = os.path.join(OUTPUT_FILES_DIR, f"xgboost_measurements_{gender}.csv")
        xgb_df.to_csv(xgb_path, index=False)
        print(f"✅ XGBoost measurements saved: {xgb_path}")

        # Save final combined measurements
        final_df = pd.DataFrame([self.final_measurements])
        final_path = os.path.join(OUTPUT_FILES_DIR, f"final_measurements_{gender}.csv")
        final_df.to_csv(final_path, index=False)
        print(f"✅ Final measurements saved: {final_path}")

        # Create summary report
        self._create_summary_report()

        return final_path

    def _create_summary_report(self):
        """Create a human-readable summary report."""
        gender = self.input_info["gender"]
        summary_path = os.path.join(
            OUTPUT_FILES_DIR, f"measurement_summary_{gender}.txt"
        )

        with open(summary_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("HYBRID MEASUREMENT SYSTEM - SUMMARY REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write("INPUT INFORMATION:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Gender: {self.input_info['gender']}\n")
            f.write(f"Weight: {self.input_info['weight_kg']} kg\n")
            f.write(f"Age: {self.input_info['age']} years\n")
            if self.input_info["stature_cm"]:
                f.write(f"Stature (provided): {self.input_info['stature_cm']} cm\n")
            f.write("\n")

            f.write("CNN EXTRACTED MEASUREMENTS:\n")
            f.write("-" * 70 + "\n")
            for key, value in self.cnn_measurements.items():
                f.write(f"{key}: {value:.2f}\n")
            f.write("\n")

            f.write("XGBOOST PREDICTED MEASUREMENTS:\n")
            f.write("-" * 70 + "\n")
            for key, value in self.xgboost_measurements.items():
                f.write(f"{key}: {value:.2f}\n")
            f.write("\n")

            f.write("REFINED MEASUREMENTS (Averaged):\n")
            f.write("-" * 70 + "\n")
            refined_keys = [
                k for k in self.final_measurements.keys() if "_refined" in k
            ]
            for key in refined_keys:
                f.write(f"{key}: {self.final_measurements[key]:.2f}\n")
            f.write("\n")

            f.write("=" * 70 + "\n")

        print(f"✅ Summary report saved: {summary_path}")


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("HYBRID CNN-XGBOOST MEASUREMENT SYSTEM")
    print("=" * 70)
    print("Combining CNN extraction with XGBoost refinement")
    print("=" * 70 + "\n")

    try:
        # Initialize system
        system = HybridMeasurementSystem()

        # Load input information
        system.load_input_info()

        # Load models
        system.load_cnn_model()
        system.load_xgboost_model()
        system.load_segmentation_model()

        # Process images
        front_img, side_img = system.load_and_process_images()

        # Run CNN extraction
        cnn_results = system.run_cnn_extraction(front_img, side_img)

        # Compute BMI
        bmi = system.compute_bmi()

        # Run XGBoost prediction
        xgb_results = system.run_xgboost_prediction(bmi)

        # Combine measurements
        final_results = system.combine_measurements()

        # Save results
        system.save_results()

        print(f"\n{'='*70}")
        print("PROCESSING COMPLETE!")
        print(f"{'='*70}")
        print(f"\nFinal measurements contain:")
        print(f"  - {len(system.cnn_measurements)} CNN measurements")
        print(f"  - {len(system.xgboost_measurements)} XGBoost predictions")
        print(f"  - 2 refined (averaged) measurements")
        print(f"\nAll results saved to: {OUTPUT_FILES_DIR}")

        return system

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    system = main()
