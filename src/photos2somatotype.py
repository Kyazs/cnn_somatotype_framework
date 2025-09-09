#!/usr/bin/env python3
"""
Photos to Somatotype Measurements (Enhanced with Literature-Based Equations)

This script extracts somatotype measurements from input images using a hybrid approach:
1. CNN extraction of basic anthropometric measurements from photos
2. MICE imputation for missing standard measurements  
3. Advanced somatotype prediction combining:
   - Trained ML models on real CAESAR data (where available)
   - Literature-based prediction equations from peer-reviewed research
   
The system predicts 8 specialized somatotype measurements:
- 4 Skinfold measurements (triceps, subscapular, suprailiac, calf)
- 2 Bone breadth measurements (humerus, femur biepicondylar)  
- 2 Specialized circumferences (arm flexed, calf)

Uses validated equations from Jackson-Pollock (1978-1980), Durnin-Womersley (1974), 
Heymsfield (1982), and other established anthropometric research.
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
from reshaper.avatar_somatotype import AvatarSomatotype

# Constants
IMG_RESIZE = 448
MODEL_NAME = "extractor_nn_model.h5"
NUM_AUG_INPUT = 20

# Import all the functions from the original photos2avatar.py
def load_model(model_name):
    '''Loads a previously trained model.'''
    try:
        return keras.models.load_model(os.path.join(MODEL_FILES_DIR, model_name))
    except FileNotFoundError as file_error:
        print(f"Error: File not found - {file_error.filename}")
        sys.exit(1)

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
    """Extracts the silhouettes from input images"""
    print("[2] Starting to extract the silhouettes from input images (front and side views)")
    start = time.time()

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

        sil_name = f"silOrig_{view}_{input_gender_glob}.png"

        if sil_name not in os.listdir(OUTPUT_FILES_DIR):
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

            # Convert mask to RGB
            rgb_mask = np.stack([mask, mask, mask], axis=2)
            masked_image = np.array(img) * (rgb_mask / 255.0)
            result_image = PIL.Image.fromarray(masked_image.astype(np.uint8))
            result_image.save(os.path.join(OUTPUT_FILES_DIR, sil_name))

        # Load and process silhouette
        dim2 = (IMG_RESIZE, IMG_RESIZE)
        sil = PIL.Image.open(os.path.join(OUTPUT_FILES_DIR, sil_name))
        sil = PIL.ImageOps.pad(sil, dim2, color="white", centering=(0.5, 0.5))
        sil = remove_transparency(sil).convert("L")

        sil = np.array(sil).astype(np.uint8)
        sil = sil.reshape(sil.shape[0], sil.shape[1], 1)

        orig_sil = np.array(sil).astype(float)
        sil_name = f"silOrig2_{view}_{input_gender_glob}.png"
        cv.imwrite(os.path.join(OUTPUT_FILES_DIR, sil_name), orig_sil)

        dim = (IMG_SIZE_4NN, IMG_SIZE_4NN)
        sil = cv.resize(sil, dim)
        sil = sil.reshape(sil.shape[0], sil.shape[1], 1)

        thresh = filters.threshold_mean(sil)
        binary_global = sil < thresh
        sil = np.array(binary_global).astype(float)
        sil = sil.reshape(sil.shape[0], sil.shape[1], 1)

        sil_name = f"sil_{view}_{input_gender_glob}.png"
        cv.imwrite(os.path.join(OUTPUT_FILES_DIR, sil_name), sil.squeeze().astype("uint8") * 255)

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

    print(f"[2] Finished extracting the silhouettes from input images in {(time.time() - start):.1f} s")
    return sil_images

def remove_transparency(im, bg_colour=(255, 255, 255)):
    """Remove transparency from an image"""
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        alpha = im.convert("RGBA").split()[-1]
        bg = PIL.Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg
    return im

def measurements_from_sil(model, input_dataX, img_input, df, stature_cm):
    """Extracts 8 measurements from the silhouettes"""
    input_img_front = np.array(img_input[0]).reshape(1, IMG_SIZE_4NN, IMG_SIZE_4NN, 1)
    input_img_side = np.array(img_input[1]).reshape(1, IMG_SIZE_4NN, IMG_SIZE_4NN, 1)

    data_gen_args_input = dict(
        featurewise_center=True,
        featurewise_std_normalization=True,
        zoom_range=[0.8, 1.2],
    )

    datagen_input = ImageDataGenerator(**data_gen_args_input)
    datagen_input.fit(input_img_front, augment=True)

    preds_input_aug_all = np.zeros((1, len(UK_MEAS)))

    input_img_front_aug_accum = []
    input_img_side_aug_accum = []

    for _ in range(NUM_AUG_INPUT):
        batch_img_front_aug = datagen_input.flow(
            input_img_front, batch_size=1, shuffle=False, seed=random.randint(0, 100),
        ).next()[0]
        
        input_img_front_aug_accum.append(batch_img_front_aug)
        batch_img_side_aug = datagen_input.flow(
            input_img_side, batch_size=1, shuffle=False, seed=random.randint(0, 100),
        ).next()[0]
        input_img_side_aug_accum.append(batch_img_side_aug)

    input_img_front_aug = np.stack(input_img_front_aug_accum)
    input_img_side_aug = np.stack(input_img_side_aug_accum)

    repeated_input_dataX = np.repeat(input_dataX, input_img_front_aug.shape[0], axis=0)

    try:
        preds_input_aug_all = model.predict(
            [repeated_input_dataX, input_img_front_aug, input_img_side_aug], verbose=0
        )
    except ValueError:
        print("Please, make sure the model file you are using is the correct one - double check the model architecture and the input data.")
        sys.exit(1)

    mean_values = df[UK_MEAS].mean()
    std_values = df[UK_MEAS].std()

    preds_input_scaled = preds_input_aug_all * std_values.values + mean_values.values
    preds_input = np.round(np.mean(preds_input_scaled, axis=0), 2)

    input9meas = np.hstack([np.array(stature_cm), preds_input])

    return input9meas

def main():
    """
    Main function to extract somatotype measurements from photos.
    """
    
    print("="*70)
    print("SOMATOTYPE MEASUREMENT EXTRACTION FROM PHOTOS")
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

    try:
        print("[3] Loading the previously trained extractor model.")
        start = time.time()
        model = load_model(MODEL_NAME)
        print(f"[3] Finished loading the previously trained extractor model in {(time.time() - start):.1f} s")
    except FileNotFoundError as file_error:
        print(f"Error: {file_error}")
        sys.exit(1)

    try:
        print(f"[3.5] Starting to extract {len(UK_MEAS)} measurements from the silhouettes.")
        start = time.time()
        extracted_measurements = measurements_from_sil(model, input_info, sil_images, df_total, stature_glob)
        print(f"[3.5] Finished extracting {len(UK_MEAS)} measurements from the silhouettes in {(time.time() - start):.1f} s")
    except ValueError:
        print("Please, check the silhouettes images size - it must be 224x224 px (IMG_SIZE_4NN in utils.py). Check input_info.")
        sys.exit(1)

    print("[4] Starting somatotype measurement prediction.")
    start = time.time()

    measurements = create_measurements_array(extracted_measurements, weightkg_glob)

    # Create somatotype measurement predictor (no 3D avatar creation)
    avatar = AvatarSomatotype(measurements, input_gender_glob, enable_somatotype=True)
    
    # Predict complete measurements including somatotype
    complete_results = avatar.predict_complete(include_somatotype=True)
    
    # Save somatotype results to output folder
    avatar.save_somatotype_results(complete_results, f"somatotype_measurements_fromImg")
    
    print(f"[4] Finished somatotype measurement prediction in {(time.time() - start):.1f} s")
    
    # Print comprehensive results summary
    print("\n" + "="*70)
    print("SOMATOTYPE MEASUREMENTS RESULTS")
    print("="*70)
    
    print(f"Subject: {input_gender_glob.title()}")
    print(f"Weight: {weightkg_glob} kg, Height: {stature_glob} cm")
    
    if complete_results.get('somatotype_measurements'):
        print(f"\nPREDICTED SOMATOTYPE MEASUREMENTS:")
        somatotype_data = complete_results['somatotype_measurements']
        confidence_data = complete_results.get('confidence_scores', {})
        
        print(f"\n   Skinfold Measurements (mm):")
        skinfolds = [
            ('triceps_skinfold_mm', 'Triceps'),
            ('subscapular_skinfold_mm', 'Subscapular'), 
            ('suprailiac_skinfold_mm', 'Suprailiac'),
            ('calf_skinfold_mm', 'Calf')
        ]
        
        for measurement, display_name in skinfolds:
            value = somatotype_data.get(measurement)
            confidence = confidence_data.get(measurement, 0)
            if value is not None:
                print(f"     • {display_name}: {value:.1f} mm (confidence: {confidence:.2f})")
            else:
                print(f"     • {display_name}: Not available")
        
        print(f"\n   Bone Breadth Measurements (cm):")
        breadths = [
            ('humerus_biepicondylar_breadth_cm', 'Humerus (elbow)'),
            ('femur_biepicondylar_breadth_cm', 'Femur (knee)')
        ]
        
        for measurement, display_name in breadths:
            value = somatotype_data.get(measurement)
            confidence = confidence_data.get(measurement, 0)
            if value is not None:
                print(f"     • {display_name}: {value:.1f} cm (confidence: {confidence:.2f})")
            else:
                print(f"     • {display_name}: Not available")
        
        print(f"\n   Specialized Circumferences (cm):")
        circumferences = [
            ('arm_circumference_flexed_cm', 'Arm (flexed)'),
            ('calf_circumference_cm', 'Calf')
        ]
        
        for measurement, display_name in circumferences:
            value = somatotype_data.get(measurement)
            confidence = confidence_data.get(measurement, 0)
            if value is not None:
                print(f"     • {display_name}: {value:.1f} cm (confidence: {confidence:.2f})")
            else:
                print(f"     • {display_name}: Not available")
    
    # Validation status
    validation_status = complete_results.get('validation_status', {})
    print(f"\nValidation Status: {validation_status.get('status', 'unknown').upper()}")
    if validation_status.get('warnings'):
        print("Warnings:")
        for warning in validation_status['warnings']:
            print(f"     - {warning}")
    
    # Performance summary
    successful_predictions = sum(1 for v in complete_results.get('somatotype_measurements', {}).values() if v is not None)
    total_measurements = 8
    success_rate = (successful_predictions / total_measurements) * 100
    
    print(f"\nPERFORMANCE SUMMARY:")
    print(f"     Successful predictions: {successful_predictions}/{total_measurements}")
    print(f"     Success rate: {success_rate:.1f}%")
    
    if complete_results.get('confidence_scores'):
        confidences = [c for c in complete_results['confidence_scores'].values() if c > 0]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            print(f"     Average confidence: {avg_confidence:.2f}")
    
    print(f"\nOUTPUT FILES:")
    print(f"     Location: {OUTPUT_FILES_DIR}")
    print(f"     Files created:")
    print(f"       • somatotype_measurements_fromImg_{input_gender_glob}.csv (detailed data)")
    print(f"       • somatotype_measurements_fromImg_{input_gender_glob}_summary.txt (readable summary)")
    
    print(f"\n" + "="*70)
    print("SOMATOTYPE MEASUREMENT EXTRACTION COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    return complete_results

if __name__ == "__main__":
    main()
