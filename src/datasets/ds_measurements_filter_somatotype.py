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
        
        # Weight: Convert from pounds to kg
        if 'weight_kg' in df_filtered.columns:
            df_filtered['weight_kg'] = df_filtered['weight_kg'] / 2.20462
            print(f"     Weight converted: lbs → kg (÷2.20462)")
        
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
            print(f"     WARNING: Weight range looks unusual: {weight_range} kg")
        else:
            print(f"     ✅ Weight range looks good: {weight_range[0]:.1f}-{weight_range[1]:.1f} kg")
    
    if 'stature_cm' in df_filtered.columns:
        height_range = (df_filtered['stature_cm'].min(), df_filtered['stature_cm'].max())
        if height_range[0] < 140 or height_range[1] > 220:
            print(f"     WARNING: Height range looks unusual: {height_range} cm")
        else:
            print(f"     ✅ Height range looks good: {height_range[0]:.1f}-{height_range[1]:.1f} cm")
    
    if 'calf_circumference' in df_filtered.columns:
        calf_range = (df_filtered['calf_circumference'].min(), df_filtered['calf_circumference'].max())
        if calf_range[0] < 20 or calf_range[1] > 60:
            print(f"     WARNING: Calf circumference range looks unusual: {calf_range} cm")
        else:
            print(f"     ✅ Calf circumference range looks good: {calf_range[0]:.1f}-{calf_range[1]:.1f} cm")
    
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

def fix_existing_datasets():
    """
    Emergency function to fix existing corrupted TOTAL files by regenerating them
    with proper unit conversions.
    """
    print("\n🚨 FIXING EXISTING CORRUPTED DATASETS 🚨")
    print("This will regenerate your measurements_TOTAL_*.csv files with correct units")
    
    # First, check if we have source files to work with
    source_files_exist = True
    for ds in DATASETS:
        for gender in GENDERS:
            source_file = os.path.join(DS_ANSUR_DIR, f"{ds}_{gender}.csv")
            if not os.path.exists(source_file):
                print(f"❌ Missing source file: {source_file}")
                source_files_exist = False
    
    if not source_files_exist:
        print("❌ Cannot fix datasets - missing source files")
        return False
    
    print("✅ Source files found - proceeding with fix...")
    
    # Backup existing corrupted files
    backup_dir = os.path.join(DS_DIR, "backup_corrupted")
    os.makedirs(backup_dir, exist_ok=True)
    
    for gender in GENDERS:
        total_file = os.path.join(DS_DIR, f"measurements_TOTAL_{gender}.csv")
        if os.path.exists(total_file):
            backup_file = os.path.join(backup_dir, f"measurements_TOTAL_{gender}_corrupted.csv")
            import shutil
            shutil.copy2(total_file, backup_file)
            print(f"📦 Backed up corrupted file to: {os.path.basename(backup_file)}")
    
    # Regenerate clean datasets
    print("\n🔧 Regenerating clean datasets...")
    generateANSURfiles()
    generateTOTALfiles()
    
    # Validate the fix
    print("\n🔍 Validating fixed datasets...")
    try:
        for gender in GENDERS:
            file_path = os.path.join(DS_DIR, f"measurements_TOTAL_{gender}.csv")
            df = pd.read_csv(file_path, encoding=FILE_ENCODING)
            
            print(f"\n{gender.upper()} dataset validation:")
            print(f"  Shape: {df.shape}")
            
            if 'calf_circumference' in df.columns:
                calf_stats = df['calf_circumference'].describe()
                print(f"  Calf circumference: {calf_stats['min']:.1f} - {calf_stats['max']:.1f} cm (mean: {calf_stats['mean']:.1f})")
                if calf_stats['mean'] > 100:
                    print("  ❌ Still corrupted - mean too high")
                    return False
                else:
                    print("  ✅ Looks good!")
            
            if 'waist_circumference' in df.columns:
                waist_stats = df['waist_circumference'].describe()
                print(f"  Waist circumference: {waist_stats['min']:.1f} - {waist_stats['max']:.1f} cm (mean: {waist_stats['mean']:.1f})")
                if waist_stats['mean'] > 200:
                    print("  ❌ Still corrupted - mean too high")
                    return False
                else:
                    print("  ✅ Looks good!")
        
        print("\n🎉 Dataset fix completed successfully!")
        print("Your model can now be retrained with clean data.")
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

# ===========================================================================

if __name__ == "__main__":
    # Ensure necessary directories exist
    os.makedirs(DS_DIR, exist_ok=True)
    
    # Check if we need to fix existing datasets
    fix_needed = False
    try:
        for gender in GENDERS:
            file_path = os.path.join(DS_DIR, f"measurements_TOTAL_{gender}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, encoding=FILE_ENCODING)
                if 'calf_circumference' in df.columns:
                    mean_calf = df['calf_circumference'].mean()
                    if mean_calf > 100:  # Clearly corrupted if mean > 100cm
                        print(f"🚨 Detected corrupted data in {gender} dataset (calf mean: {mean_calf:.1f} cm)")
                        fix_needed = True
                        break
    except:
        fix_needed = True
    
    if fix_needed:
        print("🔧 Corrupted datasets detected - running fix...")
        success = fix_existing_datasets()
        if not success:
            print("❌ Fix failed - please check source data")
    else:
        # Run normal processing pipeline
        print("🚀 Running normal data processing pipeline...")
        generateANSURfiles()
        generateTOTALfiles()
        print("\n🎉 Data processing pipeline complete!")