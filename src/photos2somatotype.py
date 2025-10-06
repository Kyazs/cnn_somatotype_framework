#!/usr/bin/env python3
"""
Photos to Anthropometric Measurements Extraction System

This script extracts anthropometric measurements from input images using a hybrid approach:
1. CNN-based extraction of basic anthropometric measurements from silhouettes
2. Literature-based estimation for missing specialized measurements
3. Physiological validation and consistency checking

The system predicts and estimates the following measurements:
- 5 CNN-extracted measurements: calf circumference, biacromial breadth, bicristal breadth, 
  waist circumference, biceps circumference (flexed)
- 4 Estimated skinfold measurements: triceps, subscapular, supraspinale, medial calf
- 2 Estimated bone breadth measurements: humerus, femur biepicondylar
- Additional derived metrics: body fat percentage, body density, fat mass

Processing Pipeline:
1. Load input photos (front and side views) and basic subject information
2. Extract silhouettes using DeepLabV3 semantic segmentation
3. Process silhouettes for CNN model input (224x224 binary images)
4. Extract measurements using trained CNN model
5. Estimate missing measurements using validated anthropometric equations
6. Generate standardized outputs in multiple formats

Uses established equations from:
- Woolcott et al. (2017) for fat mass estimation
- Durnin & Womersley (1974) for skinfold prediction
- Auerbach & Ruff (2004) for bone breadth estimation
- Siri equation for body density calculation

Note: This system is designed for research purposes and requires proper CNN model 
training with physiologically consistent data for accurate results.
"""

import os
import sys
import time
import gc
import random
import numpy as np
import pandas as pd
import PIL
import PIL.Image
import PIL.ImageOps
import cv2 as cv
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import torch
import joblib
import logging

from torchvision import transforms
from skimage import filters
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Check if any GPUs are available
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    # No GPUs available, force TensorFlow to use CPU only
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from utils import ( 
    INPUT_FILES_DIR,
    FILE_ENCODING,
    DS_DIR,
    SCALER,
    MODEL_FILES_DIR,
    CONTINUOUS,
    GENDER_DICT,
    VIEWS,
    OUTPUT_FILES_DIR,
    IMG_SIZE_4NN,
    UK_MEAS,
    M_NUM,
)

# Add the parent directory to sys.path so we can use absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import Avatar class for somatotype measurements
# from reshaper.avatar_somatotype import AvatarSomatotype

# Constants
IMG_RESIZE = 448
MODEL_NAME = "extractor_stdScal_img224_inp2_out5_ep101.keras"
NUM_AUG_INPUT = 20

# Import all the functions from the original photos2avatar.py
def load_model(model_name):
    '''Loads a previously trained model with error handling for compatibility issues.'''
    try:
        model_path = os.path.join(MODEL_FILES_DIR, model_name)
        print(f"   Attempting to load model from: {model_path}")
        
        # Try loading with custom objects scope for unknown layers
        with tf.keras.utils.custom_object_scope({'Cast': tf.cast}):
            model = keras.models.load_model(model_path, compile=False)
            print(f"   Model loaded successfully with custom objects scope")
            return model
        
    except ValueError as ve:
        if "Unknown layer" in str(ve):
            print(f"   Error: Model contains unknown layers - {ve}")
            print("   Trying alternative loading method...")
            
            # Try without custom object scope
            try:
                model = keras.models.load_model(model_path, compile=False)
                print(f"   Model loaded successfully (fallback method)")
                return model
            except Exception as e2:
                print(f"   Fallback method also failed: {e2}")
                
                # Try loading the .keras version if we were trying .h5
                if model_name.endswith('.h5'):
                    keras_name = model_name.replace('.h5', '.keras')
                    print(f"   Trying alternate format: {keras_name}")
                    try:
                        with tf.keras.utils.custom_object_scope({'Cast': tf.cast}):
                            model = keras.models.load_model(os.path.join(MODEL_FILES_DIR, keras_name), compile=False)
                            print(f"   Successfully loaded {keras_name}")
                            return model
                    except Exception as e3:
                        print(f"   Alternate format also failed: {e3}")
        raise ve
        
    except FileNotFoundError as file_error:
        print(f"Error: File not found - {file_error.filename}")
        raise file_error
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

def estimate_missing_somatotype_measurements(extracted_measurements, gender):
    """
    Estimates missing anthropometric measurements using literature-based equations.
    
    This function takes CNN-extracted basic measurements and estimates specialized
    anthropometric measurements using validated regression equations and body 
    composition models from peer-reviewed research.
    
    Args:
        extracted_measurements: dict with keys ['weight_kg', 'stature_cm', 'calf_circumference', 
                                              'biacromial_breadth', 'bicristal_breadth', 
                                              'waist_circumference', 'biceps_circumference_flexed']
        gender: 'male' or 'female'
    
    Returns:
        dict: Complete measurements including:
            - Original extracted measurements
            - Estimated bone breadths (humerus, femur) in cm
            - Estimated skinfolds (triceps, subscapular, supraspinale, medial_calf) in mm
            - Derived body composition metrics (fat mass, body fat %, body density)
            
    Method:
        1. Validates input measurements for physiological consistency
        2. Estimates bone breadths using Auerbach & Ruff (2004) regression models
        3. Estimates fat mass using NHANES model (Woolcott et al., 2017)
        4. Calculates body density using Siri equation
        5. Estimates sum of 4 skinfolds using Durnin & Womersley (1974)
        6. Apportions skinfolds to individual sites using physiological ratios
        
    Note:
        This function assumes input measurements are physiologically consistent.
        Extreme or inconsistent inputs may produce unrealistic outputs.
    """
    
    # Extract known measurements
    weight_kg = extracted_measurements['weight_kg']
    stature_cm = extracted_measurements['stature_cm']
    waist_circ_cm = extracted_measurements['waist_circumference']
    biacromial_breadth_cm = extracted_measurements['biacromial_breadth']
    bicristal_breadth_cm = extracted_measurements['bicristal_breadth']
    calf_circ_cm = extracted_measurements['calf_circumference']
    biceps_circ_flexed_cm = extracted_measurements['biceps_circumference_flexed']
    
    print(f"   Input measurements for estimation:")
    print(f"     Weight: {weight_kg} kg, Height: {stature_cm} cm")
    print(f"     Waist circumference: {waist_circ_cm} cm")
    print(f"     Biacromial breadth: {biacromial_breadth_cm} cm")
    print(f"     Bicristal breadth: {bicristal_breadth_cm} cm")
    
    # VALIDATE INPUT MEASUREMENTS FIRST
    if weight_kg <= 0 or stature_cm <= 0:
        print(f"   ERROR: Invalid weight ({weight_kg}) or height ({stature_cm})")
        return extracted_measurements
    
    if waist_circ_cm <= 0 or waist_circ_cm < 30 or waist_circ_cm > 200:
        print(f"   ERROR: Invalid waist circumference: {waist_circ_cm} cm")
        print(f"   Skipping skinfold estimation due to invalid input")
        # Return measurements with bone breadths only
        biacromial_breadth_mm = biacromial_breadth_cm * 10
        bicristal_breadth_mm = bicristal_breadth_cm * 10
        humerus_breadth_mm = 0.16 * biacromial_breadth_mm + 4.9
        femur_breadth_mm = 0.19 * bicristal_breadth_mm + 36.6
        
        return {
            **extracted_measurements,
            'humerus_breadth': humerus_breadth_mm / 10,
            'femur_breadth': femur_breadth_mm / 10,
            'triceps_skinfold': None,
            'subscapular_skinfold': None,
            'supraspinale_skinfold': None,
            'medial_calf_skinfold': None,
            'estimated_fat_mass_kg': None,
            'estimated_body_fat_percent': None,
            'estimated_body_density': None,
            'estimated_sum_4_skinfolds_mm': None
        }
    
    # STEP 1: Estimate bone breadths using regression equations from the paper
    print("   Estimating bone breadths using Auerbach & Ruff (2004) and Ruff et al. (1991) models...")
    
    # Convert to mm for equations
    biacromial_breadth_mm = biacromial_breadth_cm * 10
    bicristal_breadth_mm = bicristal_breadth_cm * 10
    
    # Humerus breadth estimation: Humerus Breadth (mm) = 0.16 × (Biacromial Breadth, mm) + 4.9
    humerus_breadth_mm = 0.16 * biacromial_breadth_mm + 4.9
    
    # Femur breadth estimation: Femur Breadth (mm) = 0.19 × (Bicristal Breadth, mm) + 36.6  
    femur_breadth_mm = 0.19 * bicristal_breadth_mm + 36.6
    
    print(f"     Humerus breadth: {humerus_breadth_mm:.2f} mm ({humerus_breadth_mm/10:.2f} cm)")
    print(f"     Femur breadth: {femur_breadth_mm:.2f} mm ({femur_breadth_mm/10:.2f} cm)")
    
    # STEP 2: Multi-step skinfold estimation process
    print("   Estimating skinfolds using multi-step body fat prediction...")
    
    # Step 2.1: Estimate Fat Mass using NHANES model (Woolcott et al., 2017)
    if gender.lower() == 'male':
        # Males: FM (kg) = 0.53(Wt) + 0.52(Waist Circ) - 0.08(Ht) - 22.46
        fat_mass_kg_raw = 0.53 * weight_kg + 0.52 * waist_circ_cm - 0.08 * stature_cm - 22.46
    else:
        # Females: FM (kg) = 0.45(Wt) + 0.88(Waist Circ) - 0.12(Ht) - 21.39  
        fat_mass_kg_raw = 0.45 * weight_kg + 0.88 * waist_circ_cm - 0.12 * stature_cm - 21.39
    
    print(f"     Raw estimated fat mass: {fat_mass_kg_raw:.2f} kg")
    
    # Ensure fat mass is positive and reasonable
    fat_mass_kg = max(fat_mass_kg_raw, 0.5)
    fat_mass_kg = min(fat_mass_kg, weight_kg * 0.5)  # Cap at 50% of body weight
    
    if fat_mass_kg != fat_mass_kg_raw:
        print(f"     WARNING: Fat mass capped from {fat_mass_kg_raw:.2f} to {fat_mass_kg:.2f} kg")
    
    # Step 2.2: Calculate % Body Fat
    body_fat_percent_raw = (fat_mass_kg / weight_kg) * 100
    body_fat_percent = max(body_fat_percent_raw, 3.0)  # Minimum essential fat
    body_fat_percent = min(body_fat_percent, 50.0)  # Increased cap to 50% for obese subjects
    
    if body_fat_percent != body_fat_percent_raw:
        print(f"     WARNING: Body fat % capped from {body_fat_percent_raw:.2f}% to {body_fat_percent:.2f}%")
    
    print(f"     Estimated fat mass: {fat_mass_kg:.2f} kg")
    print(f"     Estimated body fat %: {body_fat_percent:.2f}%")
    
    # Step 2.3: Calculate Body Density using Siri equation
    body_density = 4.95 / (body_fat_percent + 4.5)
    
    # Validate body density (should be around 1.0-1.1 g/cm³)
    if body_density < 0.9:
        print(f"   WARNING: Unrealistically low body density: {body_density:.4f} g/cm³")
        print(f"   This indicates a problem with input measurements or body composition")
        print(f"   Adjusting body density to minimum physiological value: 0.95 g/cm³")
        body_density = 0.95
    elif body_density > 1.2:
        print(f"   WARNING: Unrealistically high body density: {body_density:.4f} g/cm³")
        print(f"   Adjusting body density to maximum physiological value: 1.15 g/cm³")
        body_density = 1.15
    
    print(f"     Body density (adjusted): {body_density:.4f} g/cm³")
    
    # Step 2.4: Estimate Sum of 4 Skinfolds using Durnin & Womersley (1974)
    # Constants for age 20-29
    if gender.lower() == 'male':
        C, M = 1.1631, 0.0632
    else:
        C, M = 1.1599, 0.0717
    
    # Σ4SF (mm) = 10^((C - Db) / M)
    exponent = (C - body_density) / M
    
    print(f"     Calculated exponent: {exponent:.4f}")
    
    # Prevent extreme exponents
    if exponent > 2.5:  # 10^2.5 = 316mm, very high but possible
        print(f"   WARNING: Extreme exponent {exponent:.2f}, capping at 2.5")
        print(f"   This suggests very high body fat percentage")
        exponent = 2.5
    elif exponent < 1.0:  # 10^1.0 = 10mm, very low
        print(f"   WARNING: Low exponent {exponent:.2f}, setting to 1.0")
        print(f"   This suggests very low body fat percentage")
        exponent = 1.0
    
    sum_4_skinfolds_mm = 10 ** exponent
    
    print(f"     Calculated sum of 4 skinfolds: {sum_4_skinfolds_mm:.1f} mm")
    
    # Additional safety check (less restrictive)
    if sum_4_skinfolds_mm > 300:  # 300mm total is extremely high but possible
        print(f"   WARNING: Extremely high sum of skinfolds: {sum_4_skinfolds_mm:.1f} mm, capping at 300mm")
        sum_4_skinfolds_mm = 300.0
    elif sum_4_skinfolds_mm < 15:  # 15mm total is very low
        print(f"   WARNING: Very low sum of skinfolds: {sum_4_skinfolds_mm:.1f} mm, setting to 15mm")
        sum_4_skinfolds_mm = 15.0
    
    print(f"     Final sum of 4 skinfolds: {sum_4_skinfolds_mm:.1f} mm")
    
    # Step 2.5: Apportion to individual skinfold sites
    triceps_sf_mm = 0.30 * sum_4_skinfolds_mm
    subscapular_sf_mm = 0.35 * sum_4_skinfolds_mm  
    supraspinale_sf_mm = 0.35 * sum_4_skinfolds_mm
    medial_calf_sf_mm = 0.60 * triceps_sf_mm
    
    print(f"     Individual skinfolds (mm):")
    print(f"       Triceps: {triceps_sf_mm:.1f}")
    print(f"       Subscapular: {subscapular_sf_mm:.1f}")
    print(f"       Supraspinale: {supraspinale_sf_mm:.1f}")
    print(f"       Medial calf: {medial_calf_sf_mm:.1f}")
    
    # Create complete measurements dictionary
    complete_measurements = {
        # Original extracted measurements
        'weight_kg': weight_kg,
        'stature_cm': stature_cm,
        'calf_circumference': calf_circ_cm,
        'biacromial_breadth': biacromial_breadth_cm,
        'bicristal_breadth': bicristal_breadth_cm,
        'waist_circumference': waist_circ_cm,
        'biceps_circumference_flexed': biceps_circ_flexed_cm,
        
        # Estimated bone breadths (converted back to cm)
        'humerus_breadth': humerus_breadth_mm / 10,
        'femur_breadth': femur_breadth_mm / 10,
        
        # Estimated skinfolds (keep in mm - their natural anthropometric unit)
        'triceps_skinfold': triceps_sf_mm,
        'subscapular_skinfold': subscapular_sf_mm,
        'supraspinale_skinfold': supraspinale_sf_mm,
        'medial_calf_skinfold': medial_calf_sf_mm,
        
        # Intermediate calculations for validation
        'estimated_fat_mass_kg': fat_mass_kg,
        'estimated_body_fat_percent': body_fat_percent,
        'estimated_body_density': body_density,
        'estimated_sum_4_skinfolds_mm': sum_4_skinfolds_mm
    }
    
    return complete_measurements

def create_measurements_array(extracted_measurements, weightkg_glob):
    '''Creates an array with the measurements to be used for somatotype prediction.'''
    measurements = np.zeros(M_NUM).transpose()
    measurements[0] = weightkg_glob
    measurements[1] = extracted_measurements[0]
    measurements[3] = extracted_measurements[1]
    measurements[4] = extracted_measurements[2]
    measurements[5] = extracted_measurements[3]
    measurements[7] = extracted_measurements[4]
    measurements[14] = extracted_measurements[5]
    measurements[17] = extracted_measurements[6]
    measurements[6] = extracted_measurements[7]
    measurements[16] = extracted_measurements[8]
    
    return measurements

def get_input_data():
    """Reads and processes input data."""
    print("[1] Starting to load input data (2 photos and basic info)")
    start = time.time()

    basic_info, df_total = import_basic_info()
    input_images = load_input_images()

    print(f"[1] Finished loading input data (2 photos and basic info) in {(time.time() - start):.1f} s")
    return basic_info, input_images, df_total

def import_basic_info():
    """Imports input data (gender, height, weight)"""
    input_db_dir = os.path.join(INPUT_FILES_DIR, "input_info_extractor.csv")
    try:
        input_df = pd.read_csv(input_db_dir, encoding=FILE_ENCODING)
        global input_gender_glob
        input_gender_glob = input_df.iloc[0]["gender"]
    except FileNotFoundError:
        print(f"Wrong file or file path --> '{input_db_dir}' file not found.")
        sys.exit(1)

    input_df = input_df.replace("male", 1)
    input_df = input_df.replace("female", 0)

    tot_db_dir = os.path.join(DS_DIR, f"measurements_TOTAL_{input_gender_glob}.csv")

    try:
        df_total = pd.read_csv(tot_db_dir, encoding=FILE_ENCODING, converters={"ID": str})
    except FileNotFoundError:
        print(f"Error: File not found - '{tot_db_dir}'. Please check the file path.")
        raise

    if isinstance(SCALER, StandardScaler):
        tot_scaler_dir = os.path.join(MODEL_FILES_DIR, "scalerStd_img2ava.pkl")
    else:
        tot_scaler_dir = os.path.join(MODEL_FILES_DIR, "scalerMinMax_img2ava.pkl")

    try:
        data_measX_scaler = joblib.load(tot_scaler_dir)
    except FileNotFoundError:
        data_measX_scaler = scale_db_values()
        joblib.dump(data_measX_scaler, tot_scaler_dir)

    global stature_glob, weightkg_glob
    stature_glob = input_df.iloc[0]["stature_cm"]
    weightkg_glob = input_df.iloc[0]["weight_kg"]

    aux_df = data_measX_scaler.transform(input_df[CONTINUOUS])
    gender_aux = np.array(input_df["gender"])
    gender_aux = gender_aux.reshape(gender_aux.shape[0], -1)
    gender_cat = keras.utils.to_categorical(gender_aux, len(GENDER_DICT.keys()))
    input_dataX = np.hstack([gender_cat, aux_df])

    return input_dataX, df_total

def scale_db_values():
    """Scale the continuous values in the DataFrame using the specified scaler."""
    tot_db_dir_fem = os.path.join(DS_DIR, "measurements_TOTAL_female.csv")
    tot_db_dir_mal = os.path.join(DS_DIR, "measurements_TOTAL_male.csv")

    df_fem = pd.read_csv(tot_db_dir_fem, encoding=FILE_ENCODING, converters={"ID": str})
    df_mal = pd.read_csv(tot_db_dir_mal, encoding=FILE_ENCODING, converters={"ID": str})
    df = pd.concat([df_fem, df_mal], ignore_index=True)

    (train_data, _) = train_test_split(df, train_size=0.8, shuffle=True, random_state=123)
    scaler = StandardScaler()
    scaler.fit(train_data[CONTINUOUS])

    return scaler

def load_input_images():
    """Imports input images (front and side views)"""
    img_list = list()
    for view in VIEWS:
        filename = f"input_{view}.png"
        img = cv.imread(os.path.join(INPUT_FILES_DIR, filename), cv.IMREAD_UNCHANGED)

        if img.shape[0] > 2048:
            resized = cv.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
        else:
            resized = img

        img_list.append(PIL.Image.fromarray(np.uint8(resized)))

    return img_list

def extract_silhouette(input_images):
    """Extracts the silhouettes from input images and saves all processing steps"""
    print("[2] Starting to extract the silhouettes from input images (front and side views)")
    start = time.time()
    
    # Create timestamp for this processing session
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create subdirectory for this session's silhouette outputs
    session_dir = os.path.join(OUTPUT_FILES_DIR, f"silhouettes_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    print(f"   Saving silhouette processing steps to: {session_dir}")

    sil_images = list()

    deeplab_model = torch.hub.load("pytorch/vision:v0.10.0", "deeplabv3_resnet101", weights="DEFAULT")
    deeplab_model.eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for vi, img in enumerate(input_images):
        view = VIEWS[vi]
        print(f"** Starting {view} view")
        start_view = time.time()

        # Step 1: Save original input image
        orig_img_name = f"01_original_{view}_{input_gender_glob}_{timestamp}.png"
        img.save(os.path.join(session_dir, orig_img_name))
        print(f"   Saved original {view} image: {orig_img_name}")

        # Step 2: Generate/load segmentation mask
        sil_name = f"silOrig_{view}_{input_gender_glob}.png"
        mask_generated = False

        # Always generate fresh segmentation for current input image
        print(f"   Generating segmentation mask for {view} view...")
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            deeplab_model.to('cuda')

        with torch.no_grad():
            output = deeplab_model(input_batch)['out'][0]
        output_predictions = output.argmax(0)

        mask = output_predictions.byte().cpu().numpy()
        mask[mask != 15] = 0
        mask[mask == 15] = 255

        # Save the raw segmentation mask
        mask_name = f"02_segmentation_mask_{view}_{input_gender_glob}_{timestamp}.png"
        cv.imwrite(os.path.join(session_dir, mask_name), mask)
        print(f"   Saved segmentation mask: {mask_name}")

        # Convert mask to RGB
        rgb_mask = np.stack([mask, mask, mask], axis=2)
        masked_image = np.array(img) * (rgb_mask / 255.0)
        result_image = PIL.Image.fromarray(masked_image.astype(np.uint8))
        result_image.save(os.path.join(OUTPUT_FILES_DIR, sil_name))
        
        # Save the masked result to session directory too
        masked_name = f"03_masked_original_{view}_{input_gender_glob}_{timestamp}.png"
        result_image.save(os.path.join(session_dir, masked_name))
        print(f"   Saved masked original: {masked_name}")
        mask_generated = True

        # Step 3: Load and process silhouette
        dim2 = (IMG_RESIZE, IMG_RESIZE)
        sil = PIL.Image.open(os.path.join(OUTPUT_FILES_DIR, sil_name))
        
        # Save padded version
        sil_padded = PIL.ImageOps.pad(sil, dim2, color="white", centering=(0.5, 0.5))
        padded_name = f"04_padded_{view}_{input_gender_glob}_{timestamp}.png"
        sil_padded.save(os.path.join(session_dir, padded_name))
        print(f"   Saved padded silhouette: {padded_name}")
        
        sil = remove_transparency(sil_padded).convert("L")
        
        # Save grayscale version
        gray_name = f"05_grayscale_{view}_{input_gender_glob}_{timestamp}.png"
        sil.save(os.path.join(session_dir, gray_name))
        print(f"   Saved grayscale: {gray_name}")

        sil = np.array(sil).astype(np.uint8)
        sil = sil.reshape(sil.shape[0], sil.shape[1], 1)

        # Step 4: Save high-resolution processed version
        orig_sil = np.array(sil).astype(float)
        sil_name = f"silOrig2_{view}_{input_gender_glob}.png"
        cv.imwrite(os.path.join(OUTPUT_FILES_DIR, sil_name), orig_sil)
        
        # Save to session directory with better name
        highres_name = f"06_highres_processed_{view}_{input_gender_glob}_{timestamp}.png"
        cv.imwrite(os.path.join(session_dir, highres_name), orig_sil)
        print(f"   Saved high-res processed: {highres_name}")

        # Step 5: Resize to model input size
        dim = (IMG_SIZE_4NN, IMG_SIZE_4NN)
        sil = cv.resize(sil, dim)
        sil = sil.reshape(sil.shape[0], sil.shape[1], 1)

        # Save resized version
        resized_name = f"07_resized_{IMG_SIZE_4NN}x{IMG_SIZE_4NN}_{view}_{input_gender_glob}_{timestamp}.png"
        cv.imwrite(os.path.join(session_dir, resized_name), sil.squeeze())
        print(f"   Saved resized ({IMG_SIZE_4NN}x{IMG_SIZE_4NN}): {resized_name}")

        # Step 6: Apply threshold and create binary silhouette
        thresh = filters.threshold_mean(sil)
        binary_global = sil < thresh
        sil = np.array(binary_global).astype(float)
        sil = sil.reshape(sil.shape[0], sil.shape[1], 1)

        # Step 7: Save final binary silhouette (for model input)
        sil_name = f"sil_{view}_{input_gender_glob}.png"
        cv.imwrite(os.path.join(OUTPUT_FILES_DIR, sil_name), sil.squeeze().astype("uint8") * 255)
        
        # Save to session directory with better name
        final_name = f"08_final_binary_{view}_{input_gender_glob}_{timestamp}.png"
        cv.imwrite(os.path.join(session_dir, final_name), sil.squeeze().astype("uint8") * 255)
        print(f"   Saved final binary silhouette: {final_name}")

        sil_images.append(sil)

        # Cleanup
        if 'input_batch' in locals():
            del input_batch
        if 'output' in locals():
            del output
        if 'output_predictions' in locals():
            del output_predictions
        if 'sil' in locals():
            del sil
        plt.close('all')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print(f"-- Finished {view} view in {(time.time() - start_view):.1f} s")

    # Create processing summary
    summary_file = os.path.join(session_dir, f"processing_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Silhouette Processing Summary\n")
        f.write(f"============================\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Subject: {input_gender_glob}\n")
        f.write(f"Input Image Size: {IMG_RESIZE}x{IMG_RESIZE}\n")
        f.write(f"Model Input Size: {IMG_SIZE_4NN}x{IMG_SIZE_4NN}\n")
        f.write(f"Processing Time: {(time.time() - start):.1f} seconds\n\n")
        f.write(f"Processing Steps:\n")
        f.write(f"1. Original input images saved\n")
        f.write(f"2. DeepLabV3 segmentation applied\n")
        f.write(f"3. Masked silhouettes generated\n")
        f.write(f"4. Images padded to {IMG_RESIZE}x{IMG_RESIZE}\n")
        f.write(f"5. Converted to grayscale\n")
        f.write(f"6. Resized to {IMG_SIZE_4NN}x{IMG_SIZE_4NN} for model input\n")
        f.write(f"7. Binary threshold applied\n")
        f.write(f"8. Final silhouettes ready for CNN processing\n\n")
        f.write(f"Files saved in: {session_dir}\n")
    
    print(f"   Created processing summary: {os.path.basename(summary_file)}")
    print(f"[2] Finished extracting the silhouettes from input images in {(time.time() - start):.1f} s")
    print(f"[2] All silhouette processing steps saved to: {session_dir}")
    
    return sil_images

def save_measurements_to_csv(extracted_measurements, complete_measurements, gender):
    """
    Saves extracted and imputed measurements to CSV files in the output folder.
    
    Args:
        extracted_measurements: dict of measurements directly extracted from CNN model
        complete_measurements: dict of all measurements including estimated ones
        gender: 'male' or 'female'
    """
    
    # Create DataFrame for extracted measurements
    extracted_df = pd.DataFrame([extracted_measurements])
    extracted_df.insert(0, 'subject_id', f'fromImg_{gender}')
    extracted_df.insert(1, 'gender', gender)
    
    # Create DataFrame for complete measurements (extracted + imputed)
    complete_df = pd.DataFrame([complete_measurements])
    complete_df.insert(0, 'subject_id', f'fromImg_{gender}')
    complete_df.insert(1, 'gender', gender)
    
    # Save extracted measurements
    extracted_filename = os.path.join(OUTPUT_FILES_DIR, f"extracted_measurements_{gender}.csv")
    extracted_df.to_csv(extracted_filename, index=False, encoding='utf-8')
    print(f"   Saved extracted measurements to: {extracted_filename}")
    
    # Save complete measurements (extracted + imputed)
    imputed_filename = os.path.join(OUTPUT_FILES_DIR, f"imputed_measurements_{gender}.csv") 
    complete_df.to_csv(imputed_filename, index=False, encoding='utf-8')
    print(f"   Saved complete measurements to: {imputed_filename}")
    
    # Create standardized anthropometric data file
    anthro_filename = os.path.join(OUTPUT_FILES_DIR, "output_anthropometric_data.csv")
    
    # Define measurement mapping with proper units and formatting
    measurement_mapping = [
        # Basic measurements
        ('Height', 'stature_cm', 'cm', 1),
        ('Weight', 'weight_kg', 'kg', 1),
        
        # Skinfold measurements (already in mm, no conversion needed)
        ('Triceps_Skinfold', 'triceps_skinfold', 'mm', 1),
        ('Subscapular_Skinfold', 'subscapular_skinfold', 'mm', 1),
        ('Supraspinale_Skinfold', 'supraspinale_skinfold', 'mm', 1),
        ('Calf_Skinfold', 'medial_calf_skinfold', 'mm', 1),
        
        # Bone breadths (in cm)
        ('Humerus_Breadth', 'humerus_breadth', 'cm', 1),
        ('Femur_Breadth', 'femur_breadth', 'cm', 1),
        
        # Circumferences (in cm)
        ('Arm_Circumference_Flexed', 'biceps_circumference_flexed', 'cm', 1),
        ('Calf_Circumference', 'calf_circumference', 'cm', 1),
        
        # Additional circumferences
        ('Waist_Circumference', 'waist_circumference', 'cm', 1),
        ('Biacromial_Breadth', 'biacromial_breadth', 'cm', 1),
        ('Bicristal_Breadth', 'bicristal_breadth', 'cm', 1),
    ]
    
    # Create standardized data rows
    anthro_data = []
    
    for display_name, measurement_key, unit, multiplier in measurement_mapping:
        if measurement_key in complete_measurements:
            value = complete_measurements[measurement_key]
            if value is not None and isinstance(value, (int, float)):
                # Apply unit conversion (e.g., cm to mm for skinfolds)
                converted_value = value * multiplier
                
                # Format based on measurement type
                if unit == 'mm' or 'Skinfold' in display_name:
                    formatted_value = f"{converted_value:.0f}"  # Whole numbers for skinfolds
                elif unit == 'kg':
                    formatted_value = f"{converted_value:.1f}"  # 1 decimal for weight
                else:
                    formatted_value = f"{converted_value:.1f}"  # 1 decimal for other measurements
                
                anthro_data.append({
                    'Measurement': display_name,
                    'Value': formatted_value,
                    'Unit': unit
                })
            else:
                # Handle missing/null values
                anthro_data.append({
                    'Measurement': display_name,
                    'Value': 'N/A',
                    'Unit': unit
                })
    
    # Create DataFrame and save
    anthro_df = pd.DataFrame(anthro_data)
    anthro_df.to_csv(anthro_filename, index=False, encoding='utf-8')
    print(f"   Saved standardized anthropometric data to: {anthro_filename}")
    
    # Create a summary report
    summary_filename = os.path.join(OUTPUT_FILES_DIR, f"measurements_summary_{gender}.txt")
    
    with open(summary_filename, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("ANTHROPOMETRIC MEASUREMENTS EXTRACTION SUMMARY\n") 
        f.write("="*70 + "\n\n")
        
        f.write(f"Subject: {gender.title()}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("EXTRACTED MEASUREMENTS (from CNN Model):\n")
        f.write("-" * 50 + "\n")
        for key, value in extracted_measurements.items():
            if isinstance(value, float):
                f.write(f"  {key:30}: {value:8.2f}\n")
            else:
                f.write(f"  {key:30}: {value}\n")
        
        f.write("\nESTIMATED/IMPUTED MEASUREMENTS:\n") 
        f.write("-" * 50 + "\n")
        estimated_keys = [k for k in complete_measurements.keys() if k not in extracted_measurements.keys()]
        for key in estimated_keys:
            value = complete_measurements[key]
            if isinstance(value, float):
                f.write(f"  {key:30}: {value:8.2f}\n")
            else:
                f.write(f"  {key:30}: {value}\n")
        
        f.write("\nSTANDARDIZED ANTHROPOMETRIC DATA:\n")
        f.write("-" * 50 + "\n")
        f.write("The following measurements are saved in standardized format:\n")
        for row in anthro_data:
            f.write(f"  {row['Measurement']:25}: {row['Value']:8} {row['Unit']}\n")
        
        f.write("\nMETHODOLOGY SUMMARY:\n")
        f.write("-" * 50 + "\n")
        f.write("• Bone breadths estimated using Auerbach & Ruff (2004) regression models\n")
        f.write("• Skinfolds estimated using multi-step body fat prediction approach:\n")
        f.write("  - Fat mass estimated using NHANES model (Woolcott et al., 2017)\n")
        f.write("  - Body density calculated using Siri equation\n")
        f.write("  - Sum of 4 skinfolds estimated using Durnin & Womersley (1974)\n")
        f.write("  - Individual skinfolds apportioned using physiological ratios\n")
        
        f.write("\nFILE FORMAT NOTES:\n")
        f.write("-" * 50 + "\n")
        f.write("• output_anthropometric_data.csv: Standardized format with proper units\n")
        f.write("• Skinfold measurements stored in mm (anthropometric standard)\n")
        f.write("• Breadths and circumferences in cm\n")
        f.write("• Missing values marked as 'N/A'\n")
        
    print(f"   Saved measurement summary to: {summary_filename}")
    
    return extracted_filename, imputed_filename, summary_filename, anthro_filename

def remove_transparency(im, bg_colour=(255, 255, 255)):
    """Remove transparency from an image"""
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        alpha = im.convert("RGBA").split()[-1]
        bg = PIL.Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg
    return im

def measurements_from_sil(model, input_dataX, img_input, df_total, stature_cm):
    """Extracts 5 measurements from the silhouettes using the trained CNN model"""
    input_img_front = np.array(img_input[0]).reshape(1, IMG_SIZE_4NN, IMG_SIZE_4NN, 1)
    input_img_side = np.array(img_input[1]).reshape(1, IMG_SIZE_4NN, IMG_SIZE_4NN, 1)

    data_gen_args_input = dict(
        featurewise_center=True,
        featurewise_std_normalization=True,
        zoom_range=[0.8, 1.2],
    )

    datagen_input = ImageDataGenerator(**data_gen_args_input)
    datagen_input.fit(input_img_front, augment=True)

    input_img_front_aug_accum = []
    input_img_side_aug_accum = []

    for _ in range(NUM_AUG_INPUT):
        batch_img_front_aug = next(datagen_input.flow(
            input_img_front, batch_size=1, shuffle=False, seed=random.randint(0, 100),
        ))[0]
        
        input_img_front_aug_accum.append(batch_img_front_aug)
        batch_img_side_aug = next(datagen_input.flow(
            input_img_side, batch_size=1, shuffle=False, seed=random.randint(0, 100),
        ))[0]
        input_img_side_aug_accum.append(batch_img_side_aug)

    input_img_front_aug = np.stack(input_img_front_aug_accum)
    input_img_side_aug = np.stack(input_img_side_aug_accum)

    repeated_input_dataX = np.repeat(input_dataX, input_img_front_aug.shape[0], axis=0)

    try:
        # Model outputs NORMALIZED predictions
        preds_input_aug_all_normalized = model.predict(
            [repeated_input_dataX, input_img_front_aug, input_img_side_aug], verbose=0
        )
    except ValueError:
        print("Please, make sure the model file you are using is the correct one - double check the model architecture and the input data.")
        sys.exit(1)

    # Average the augmented predictions (still normalized)
    preds_normalized = np.mean(preds_input_aug_all_normalized, axis=0)
    
    # CRITICAL FIX: Load the SAME statistics used during training
    print("   Loading training statistics for proper denormalization...")
    
    # Load both male and female data (same as training)
    tot_db_dir_fem = os.path.join(DS_DIR, "measurements_TOTAL_female.csv")
    tot_db_dir_mal = os.path.join(DS_DIR, "measurements_TOTAL_male.csv")
    
    try:
        df_fem = pd.read_csv(tot_db_dir_fem, encoding=FILE_ENCODING, converters={"ID": str})
        df_mal = pd.read_csv(tot_db_dir_mal, encoding=FILE_ENCODING, converters={"ID": str})
        
        # Combine exactly like in training
        df_total_training = pd.concat([df_fem, df_mal], ignore_index=True)
        
        # Calculate the EXACT same statistics used in training
        uk_meas_mean = df_total_training[UK_MEAS].mean().values
        uk_meas_std = df_total_training[UK_MEAS].std().values
        
        print(f"   UK_MEAS columns: {UK_MEAS}")
        print(f"   Training means: {uk_meas_mean}")
        print(f"   Training stds: {uk_meas_std}")
        print(f"   Normalized predictions: {preds_normalized}")
        
        # Denormalize: real_value = (normalized_value * std) + mean
        preds_real = (preds_normalized * uk_meas_std) + uk_meas_mean
        preds_real = np.round(preds_real, 2)
        
        print(f"   Denormalized predictions: {preds_real}")
        
        # Sanity check - ensure values are reasonable
        for i, pred in enumerate(preds_real):
            meas_name = UK_MEAS[i]
            if pred < 0:
                print(f"   WARNING: Negative value for {meas_name}: {pred}")
            elif meas_name == 'calf_circumference' and (pred < 20 or pred > 60):
                print(f"   WARNING: Unrealistic calf circumference: {pred} cm")
            elif meas_name == 'waist_circumference' and (pred < 50 or pred > 150):
                print(f"   WARNING: Unrealistic waist circumference: {pred} cm")
        
    except Exception as e:
        print(f"   ERROR loading training statistics: {e}")
        print("   Model predictions may be incorrect!")
        preds_real = preds_normalized  # Fallback to normalized values

    # Map predictions to measurement names  
    extracted_measurements = {
        'weight_kg': weightkg_glob,  # From input
        'stature_cm': stature_cm,    # From input
        'calf_circumference': float(preds_real[0]),
        'biacromial_breadth': float(preds_real[1]), 
        'bicristal_breadth': float(preds_real[2]),
        'waist_circumference': float(preds_real[3]),
        'biceps_circumference_flexed': float(preds_real[4])
    }

    return extracted_measurements

def main():
    """
    Main function to extract anthropometric measurements from photos.
    
    Processing workflow:
    1. Load input data (2 photos: front/side views + basic subject info)
    2. Extract silhouettes using DeepLabV3 semantic segmentation
    3. Process silhouettes into binary images for CNN model input
    4. Extract 5 basic measurements using trained CNN model
    5. Estimate additional measurements using literature-based equations
    6. Validate physiological consistency of results
    7. Save results in multiple standardized formats
    
    Output files:
    - extracted_measurements_{gender}.csv: Raw CNN model outputs
    - imputed_measurements_{gender}.csv: Complete measurement set
    - measurements_summary_{gender}.txt: Human-readable report
    - output_anthropometric_data.csv: Standardized format with proper units
    - silhouettes_{timestamp}/: Complete silhouette processing pipeline
    
    Returns:
        dict: Processing results including measurements and output file paths
        
    Note:
        Requires properly trained CNN model and physiologically consistent input data
        for accurate results. Current model may need retraining for optimal performance.
    """
    
    print("="*70)
    print("ANTHROPOMETRIC MEASUREMENT EXTRACTION FROM PHOTOS")
    print("Using CNN-based silhouette analysis with literature-based estimation")
    print("="*70)

    try:
        input_info, input_images, df_total = get_input_data()
        sil_images = extract_silhouette(input_images)
    except (FileNotFoundError, IOError) as error:
        print(f"Error: {error}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

    # Load the trained model
    print("[3] Loading trained CNN model for measurement extraction...")
    try:
        start = time.time()
        model = load_model(MODEL_NAME)
        print(f"   Successfully loaded model: {MODEL_NAME} in {(time.time() - start):.1f} s")
    except FileNotFoundError as file_error:
        print(f"Error: {file_error}")
        sys.exit(1)

    # Extract measurements using CNN model
    print("[3.5] Extracting measurements from silhouettes using CNN model...")
    try:
        start = time.time()
        extracted_measurements = measurements_from_sil(model, input_info, sil_images, df_total, stature_glob)
        print(f"   Successfully extracted {len(extracted_measurements)} measurements in {(time.time() - start):.1f} s")
    except ValueError:
        print("Please, check the silhouettes images size - it must be 224x224 px (IMG_SIZE_4NN in utils.py). Check input_info.")
        sys.exit(1)

    # Estimate missing measurements using literature-based methods
    print("[4] Estimating missing anthropometric measurements using literature-based methods...")
    start = time.time()
    
    complete_measurements = estimate_missing_somatotype_measurements(extracted_measurements, input_gender_glob)
    
    print(f"[4] Finished anthropometric measurement estimation in {(time.time() - start):.1f} s")
    
    # Save results to CSV files
    print("[5] Saving measurements to output files...")
    extracted_file, imputed_file, summary_file, anthro_file = save_measurements_to_csv(
        extracted_measurements, complete_measurements, input_gender_glob
    )
    
    # Print comprehensive results summary
    print("\n" + "="*70)
    print("ANTHROPOMETRIC MEASUREMENTS RESULTS")
    print("="*70)
    
    print(f"Subject: {input_gender_glob.title()}")
    print(f"Weight: {weightkg_glob} kg, Height: {stature_glob} cm")
    
    print(f"\nEXTRACTED MEASUREMENTS (CNN Model):")
    print("-" * 40)
    for key, value in extracted_measurements.items():
        if isinstance(value, (int, float)):
            print(f"  {key:25}: {value:8.2f}")
        else:
            print(f"  {key:25}: {value}")
    
    print(f"\nESTIMATED MEASUREMENTS (Literature-based):")
    print("-" * 40) 
    estimated_keys = [k for k in complete_measurements.keys() if k not in extracted_measurements.keys()]
    for key in estimated_keys:
        value = complete_measurements[key]
        if isinstance(value, (int, float)):
            print(f"  {key:25}: {value:8.2f}")
        else:
            print(f"  {key:25}: {value}")
    
    # Performance summary
    total_extracted = len(extracted_measurements) 
    total_estimated = len(estimated_keys)
    total_measurements = total_extracted + total_estimated
    
    print(f"\nPERFORMANCE SUMMARY:")
    print(f"     Extracted measurements: {total_extracted}")
    print(f"     Estimated measurements: {total_estimated}")
    print(f"     Total measurements: {total_measurements}")
    
    print(f"\nOUTPUT FILES:")
    print(f"     Location: {OUTPUT_FILES_DIR}")
    print(f"     Files created:")
    print(f"       • {os.path.basename(extracted_file)} (CNN extracted measurements)")
    print(f"       • {os.path.basename(imputed_file)} (complete measurements)")
    print(f"       • {os.path.basename(summary_file)} (readable summary)")
    print(f"       • {os.path.basename(anthro_file)} (standardized anthropometric data)")
    
    print(f"\n" + "="*70)
    print("ANTHROPOMETRIC MEASUREMENT EXTRACTION COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    return {
        'extracted_measurements': extracted_measurements,
        'complete_measurements': complete_measurements,
        'output_files': {
            'extracted': extracted_file,
            'complete': imputed_file, 
            'summary': summary_file,
            'anthropometric': anthro_file
        }
    }

if __name__ == "__main__":
    main()
