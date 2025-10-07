""" 
This script filters the CAESAR dataset for the measurements needed for predicting skinfold model 
and the Ansur dataset for the same measurements, and saves the filtered datasets.


needed measurements from ANSURI:

STATURE, WEIGHT, CHEST_CIRC, BUTTOCK_CIRC, WAIST_CIRC_NATURAL, THIGH_CIRC-PROXIMAL, ANKLE_CIRC, BIACROMIAL_BRTH, KNEE_HT_-_SITTING,CALF_CIRC, CALF_CIRC, ARMCIRCBCPS_FLEX



needed measurements from ANSURII:
stature, weightkg, chestcircumference, buttockcircumference, waistcircumference, thighcircumference, anklecircumference, biacromialbreadth, kneeheightsitting, calfcircumference, bicepscircumferenceflexed



"""

import pandas as pd
import numpy as np
import os
import sys

# Append the path to your utils file if it's in a parent directory
sys.path.append("..") 
from utils import * # Assumes GENDERS, DATASETS, FILE_ENCODING are in utils.py

DS_DIR = "../../data/datasets/"
DS_ANSUR_DIR = os.path.join(DS_DIR, "ds_ansur_original")

# --- Configuration ---

# 1. Define the final, standardized column names you want in your dataset.
MEASUREMENTS = [
    "stature_cm",
    "weight_kg",
    "chest_circumference",
    "buttock_circumference",
    "waist_circumference",
    "thigh_circumference",
    "ankle_circumference",
    "biacromial_breadth",
    "knee_height_sitting",
    "calf_circumference",
    "arm_circumference_flexed",
]

# 2. List the original column names from each raw dataset that you need to keep.
#    The order MUST match the MEASUREMENTS list above.
ANSURI_COLS = [
    "STATURE",
    "WEIGHT",
    "CHEST_CIRC",
    "BUTTOCK_CIRC",
    "WAIST_CIRC_NATURAL",
    "THIGH_CIRC-PROXIMAL",
    "ANKLE_CIRC",
    "BIACROMIAL_BRTH",
    "KNEE_HT_-_SITTING",
    "CALF_CIRC",
    "ARMCIRCBCPS_FLEX",
]

ANSURII_COLS = [
    "stature",
    "weightkg",
    "chestcircumference",
    "buttockcircumference",
    "waistcircumference",
    "thighcircumference",
    "anklecircumference",
    "biacromialbreadth",
    "kneeheightsitting",
    "calfcircumference",
    "bicepscircumferenceflexed"
]

# 3. Create dictionaries to map the original names to your new standardized names.
#    This is more robust than relying on list order.
NEW_NAMES_DICT_ANSURI = dict(zip(ANSURI_COLS, MEASUREMENTS))
NEW_NAMES_DICT_ANSURII = dict(zip(ANSURII_COLS, MEASUREMENTS))

# --- Main Functions ---

def generateANSURfiles():
    """
    Loads raw ANSUR I and ANSUR II files, filters them to keep only required
    measurements, standardizes column names, and saves them as new CSVs.
    """
    print("--- Generating Individual ANSUR Files for Skinfold Model ---")
    for ds in DATASETS:
        for gender in GENDERS:
            try:
                # Load the raw dataset
                raw_df = load_ds(ds, gender)
                
                # Filter, rename, and process the data
                filtered_df = filter_ds(raw_df, ds)

                # Save the cleaned data to a new CSV file
                file_path = os.path.join(DS_DIR, f"measurements_{ds}_{gender}.csv")
                filtered_df.to_csv(file_path, index=False)
                print(f"✅ Generated: {os.path.basename(file_path)}")
                
            except FileNotFoundError as e:
                print(f"❌ ERROR: Could not find file for {ds}_{gender}: {e}")
            except Exception as e:
                print(f"❌ ERROR processing {ds}_{gender}: {e}")
                raise

def load_ds(ds, gender):
    """Loads a single raw CSV dataset."""
    ds_dir = os.path.join(DS_ANSUR_DIR, f"{ds}_{gender}.csv")
    print(f"Loading raw file: {os.path.basename(ds_dir)}")
    df = pd.read_csv(ds_dir, encoding=FILE_ENCODING, converters={"ID": str})
    return df

def filter_ds(df, ds):
    """
    Filters the dataframe to keep only necessary columns, renames them,
    and performs unit conversions for data normalization.
    """
    if ds == "ANSURI":
        cols_to_keep = ANSURI_COLS
        rename_dict = NEW_NAMES_DICT_ANSURI
    elif ds == "ANSURII":
        cols_to_keep = ANSURII_COLS
        rename_dict = NEW_NAMES_DICT_ANSURII
    else:
        # Handle cases where the dataset is not ANSURI or ANSURII
        return pd.DataFrame() 

    # 1. Select only the columns we need from the original dataframe
    #    Use .copy() to prevent SettingWithCopyWarning later.
    df_filtered = df[cols_to_keep].copy()

    # 2. Rename the columns to the standardized names
    df_filtered.rename(columns=rename_dict, inplace=True)
    
    # 3. Apply unit conversions according to METHOD FOR THE IMPOSSIBLE requirements
    print(f"   Processing {ds} - applying unit conversions...")
    
    if ds == "ANSURI":
        # ANSUR I conversions
        print(f"     Original sample (first row): {df_filtered.iloc[0].to_dict()}")
        
        # Weight: Convert from hectograms to kg
        if 'weight_kg' in df_filtered.columns:
            df_filtered['weight_kg'] = df_filtered['weight_kg'] / 10.0
            print(f"     Weight converted: hectograms → kg (÷10)")
        
        # All measurements in ANSUR I are in mm, convert to cm
        measurement_cols = [col for col in df_filtered.columns if col not in ['weight_kg']]
        for col in measurement_cols:
            df_filtered[col] = df_filtered[col] / 10.0
        print(f"     Measurements converted: mm → cm (÷10)")
        
    elif ds == "ANSURII":
        # ANSUR II conversions
        print(f"     Original sample (first row): {df_filtered.iloc[0].to_dict()}")
        
        # Weight: Convert from kg*10 to kg
        if 'weight_kg' in df_filtered.columns:
            df_filtered['weight_kg'] = df_filtered['weight_kg'] / 10.0
            print(f"     Weight converted: kg*10 → kg (÷10)")
        
        # All other measurements in ANSUR II are in mm, convert to cm
        measurement_cols = [col for col in df_filtered.columns if col not in ['weight_kg']]
        for col in measurement_cols:
            df_filtered[col] = df_filtered[col] / 10.0
        print(f"     Measurements converted: mm → cm (÷10)")
    
    # 4. Validate the conversions
    print(f"     Converted sample (first row): {df_filtered.iloc[0].to_dict()}")
    
    # 5. Basic sanity checks
    if 'weight_kg' in df_filtered.columns:
        weight_range = (df_filtered['weight_kg'].min(), df_filtered['weight_kg'].max())
        if weight_range[0] < 30 or weight_range[1] > 200:
            print(f"     WARNING: Weight range looks unusual: {weight_range[0]:.1f}-{weight_range[1]:.1f} kg")
        else:
            print(f"     ✅ Weight range looks good: {weight_range[0]:.1f}-{weight_range[1]:.1f} kg")
    
    if 'stature_cm' in df_filtered.columns:
        height_range = (df_filtered['stature_cm'].min(), df_filtered['stature_cm'].max())
        if height_range[0] < 140 or height_range[1] > 220:
            print(f"     WARNING: Height range looks unusual: {height_range} cm")
        else:
            print(f"     ✅ Height range looks good: {height_range[0]:.1f}-{height_range[1]:.1f} cm")
    
    if 'chest_circumference' in df_filtered.columns:
        chest_range = (df_filtered['chest_circumference'].min(), df_filtered['chest_circumference'].max())
        if chest_range[0] < 60 or chest_range[1] > 150:
            print(f"     WARNING: Chest circumference range looks unusual: {chest_range} cm")
        else:
            print(f"     ✅ Chest circumference range looks good: {chest_range[0]:.1f}-{chest_range[1]:.1f} cm")
    
    if 'waist_circumference' in df_filtered.columns:
        waist_range = (df_filtered['waist_circumference'].min(), df_filtered['waist_circumference'].max())
        if waist_range[0] < 50 or waist_range[1] > 150:
            print(f"     WARNING: Waist circumference range looks unusual: {waist_range} cm")
        else:
            print(f"     ✅ Waist circumference range looks good: {waist_range[0]:.1f}-{waist_range[1]:.1f} cm")

    return df_filtered

def generateTOTALfiles():
    """
    Loads the processed individual files, saves them as .npy, and concatenates
    them into TOTAL files for each gender.
    """
    print("\n--- Generating TOTAL and NPY Files for Skinfold Model ---")
    for gender in GENDERS:
        try:
            total_data = [] # List to store DataFrames for concatenation

            # Process ANSURI
            file_path_ansurI = os.path.join(DS_DIR, f"measurements_ANSURI_{gender}.csv")
            df_ansurI = pd.read_csv(file_path_ansurI, encoding=FILE_ENCODING)
            total_data.append(df_ansurI)
            np.save(os.path.join(DS_DIR, f"measurements_ANSURI_{gender}.npy"), df_ansurI.to_numpy())
            print(f"✅ Generated NPY: measurements_ANSURI_{gender}.npy")

            # Process ANSURII
            file_path_ansurII = os.path.join(DS_DIR, f"measurements_ANSURII_{gender}.csv")
            df_ansurII = pd.read_csv(file_path_ansurII, encoding=FILE_ENCODING)
            total_data.append(df_ansurII)
            np.save(os.path.join(DS_DIR, f"measurements_ANSURII_{gender}.npy"), df_ansurII.to_numpy())
            print(f"✅ Generated NPY: measurements_ANSURII_{gender}.npy")

            # Concatenate all data for the current gender
            total_dataframe = pd.concat(total_data, ignore_index=True)

            # Save the final concatenated DataFrame to CSV
            file_path_csv = os.path.join(DS_DIR, f"measurements_TOTAL_{gender}.csv")
            total_dataframe.to_csv(file_path_csv, index=False, encoding=FILE_ENCODING)

            # Save the final concatenated DataFrame to NPY
            file_path_npy = os.path.join(DS_DIR, f"measurements_TOTAL_{gender}.npy")
            np.save(file_path_npy, total_dataframe.to_numpy())
            
            print(f"✅ Generated TOTAL files for {gender}: CSV and NPY")
        except FileNotFoundError as e:
            print(f"❌ ERROR: {e}. Please ensure generateANSURfiles() ran successfully first.")
            raise
        except Exception as e:
            print(f"❌ ERROR processing TOTAL files for {gender}: {e}")
            raise

# ===========================================================================

if __name__ == "__main__":
    # Ensure necessary directories exist
    os.makedirs(DS_DIR, exist_ok=True)
    
    # Run normal processing pipeline
    print("🚀 Running skinfold model data processing pipeline...")
    generateANSURfiles()
    generateTOTALfiles()
    print("\n🎉 Skinfold model data processing pipeline complete!") 