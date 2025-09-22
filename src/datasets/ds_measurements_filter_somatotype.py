import pandas as pd
import numpy as np
import os
import sys

# Append the path to your utils file if it's in a parent directory
sys.path.append("..") 
from utils import * # Assumes GENDERS, DATASETS, FILE_ENCODING are in utils.py

DS_DIR = "../../data/datasets/"
DS_ANSUR_DIR = os.path.join(DS_DIR, "ds_ansur_original")
DS_SPRING_DIR = os.path.join(DS_DIR, "ds_SPRING")

# --- Configuration ---

# 1. Define the final, standardized column names you want in your dataset.
#    This includes your UK_MEAS and any KN_MEAS like weight and stature.
MEASUREMENTS = [
    "weight_kg",
    "stature_cm",
    "calf_circumference",
    "biacromial_breadth",
    "bicristal_breadth",
    "waist_circumference",
    "biceps_circumference_flexed",
]

# 2. List the original column names from each raw dataset that you need to keep.
#    The order MUST match the MEASUREMENTS list above.
ANSURI_COLS = [
    "WEIGHT",
    "STATURE",
    "CALF_CIRC",
    "BIACROMIAL_BRTH",
    "BISPINOUS_BRTH",
    "WAIST_CIRC-OMPHALION",
    "ARMCIRCBCPS_FLEX",
]

ANSURII_COLS = [
    "weightkg",
    "stature",
    "calfcircumference",
    "biacromialbreadth",
    "bicristalbreadth",
    "waistcircumference",
    "bicepscircumferenceflexed",
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
    print("--- Generating Individual ANSUR Files ---")
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
    and performs unit conversions.
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
    
    # 3. For ANSUR II, convert units from mm to cm and handle weight (kg*10 -> kg)
    if ds == "ANSURII":
        # Weight is in kg*10, so divide by 10 to get kg
        if 'weight' in df_filtered.columns:
            df_filtered['weight'] = df_filtered['weight'] / 10.0
        
        # All other measurements are in mm, so divide by 10 to get cm
        measurement_cols = [col for col in df_filtered.columns if col != 'weight']
        df_filtered[measurement_cols] = df_filtered[measurement_cols] / 10.0

    return df_filtered

def generateTOTALfiles():
    """
    Loads the processed individual files, saves them as .npy, and concatenates
    them into TOTAL files for each gender.
    """
    print("\n--- Generating TOTAL and NPY Files ---")
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
    
    # Run the processing pipeline
    generateANSURfiles()
    generateTOTALfiles()
    print("\n🚀 Data processing pipeline complete!")