import pandas as pd

def clean_and_split_anthropometric_data(input_file):
    """
    Remove specified columns, filter to essential somatotype measurements, 
    filter out rows with missing measurements, and split by gender into separate files
    
    Parameters:
    input_file (str): Path to the input CSV file
    
    Returns:
    tuple: (male_df, female_df) - Cleaned dataframes for each gender
    """
    
    # Load the data
    # Load the data with error handling for malformed CSV
    try:
        df = pd.read_csv(input_file, on_bad_lines='skip', low_memory=False)
    except Exception as e:
        print(f"Error reading CSV with default settings: {e}")
        print("Trying alternative CSV reading methods...")
        try:
            # Try with different separator
            df = pd.read_csv(input_file, sep=None, engine='python', on_bad_lines='skip', low_memory=False)
        except Exception as e2:
            print(f"Error with alternative method: {e2}")
            raise ValueError(f"Unable to read the CSV file: {input_file}")
    print(f"Original data shape: {df.shape}")
    print(f"Gender distribution:")
    if 'Gender' in df.columns:
        print(df['Gender'].value_counts())
    
    # Columns to remove (but keep Gender temporarily for splitting)
    columns_to_remove = [
        'Subject Number',
        'Recorder',
        'Measurer',
        'Age',
        'Race'
    ]
    
    # Essential measurements for somatotype imputation (filtered list)
    essential_measurements = [
        'Stature', 'Weight', 'BMI', 'Shoulder Breadth', 'Armscye Circumference',
        'Hip Breadth, Sitting', 'Hip Circumference, Maximum', 'Waist Circumference, Pref',
        'Chest Circumference', 'Thigh Circumference', 'Ankle Circumference',
        'Crotch Height', 'Knee Height', 'Arm Length (Spine to Wrist)',
        'Triceps Skinfold', 'Subscapular Skinfold', 'SH/S', 'Waist/Hip', 'Chest/Waist'
    ]
    
    # First, remove specified columns (except Gender)
    columns_to_remove_existing = [col for col in columns_to_remove if col in df.columns]
    df_temp = df.drop(columns=columns_to_remove_existing)
    
    print(f"Removed {len(columns_to_remove_existing)} columns: {columns_to_remove_existing}")
    print(f"Shape after removing columns: {df_temp.shape}")
    
    # Check for Gender column
    if 'Gender' not in df_temp.columns:
        raise ValueError("Gender column not found. Cannot split by gender.")
    
    # Split by gender first
    male_df = df_temp[df_temp['Gender'] == 'Male'].copy()
    female_df = df_temp[df_temp['Gender'] == 'Female'].copy()
    
    print(f"\nGender split:")
    print(f"Male subjects: {len(male_df)}")
    print(f"Female subjects: {len(female_df)}")
    
    # Function to clean measurements for each gender
    def clean_measurements(df_gender, gender_name):
        print(f"\nProcessing {gender_name} data:")
        
        # Filter to only essential measurements (plus Gender for now)
        available_essential = [col for col in essential_measurements if col in df_gender.columns]
        df_filtered = df_gender[['Gender'] + available_essential].copy()
        
        print(f"Filtered to {len(available_essential)} essential measurements")
        print(f"Essential measurements kept: {available_essential}")
        
        # Check for missing essential measurements
        missing_essential = [col for col in essential_measurements if col not in df_gender.columns]
        if missing_essential:
            print(f"Warning: The following essential measurements are not found in {gender_name} data: {missing_essential}")
        
        # Remove rows with missing values in essential measurements
        if available_essential:
            rows_before = len(df_filtered)
            df_cleaned = df_filtered.dropna(subset=available_essential)
            rows_after = len(df_cleaned)
            rows_removed = rows_before - rows_after
            
            print(f"Removed {rows_removed} {gender_name.lower()} rows with missing measurements")
            print(f"{gender_name} data shape after cleaning: {df_cleaned.shape}")
        else:
            df_cleaned = df_filtered.copy()
        
        # Remove Gender column now
        if 'Gender' in df_cleaned.columns:
            df_cleaned = df_cleaned.drop(columns=['Gender'])
        
        return df_cleaned
    
    # Clean measurements for each gender
    male_cleaned = clean_measurements(male_df, 'Male')
    female_cleaned = clean_measurements(female_df, 'Female')
    
    # Save to separate files
    male_filename = 'measurements_CAESAR_male_somatotype.csv'
    female_filename = 'measurements_CAESAR_female_somatotype.csv'
    
    male_cleaned.to_csv(male_filename, index=False)
    female_cleaned.to_csv(female_filename, index=False)
    
    print(f"\n" + "="*60)
    print("FILES CREATED SUCCESSFULLY:")
    print("="*60)
    print(f"Male data: {male_filename}")
    print(f"  - Shape: {male_cleaned.shape}")
    print(f"  - Columns: {len(male_cleaned.columns)}")
    
    print(f"\nFemale data: {female_filename}")
    print(f"  - Shape: {female_cleaned.shape}")
    print(f"  - Columns: {len(female_cleaned.columns)}")
    
    # Show all column names
    print(f"\nColumns in final datasets: {list(male_cleaned.columns)}")
    
    return male_cleaned, female_cleaned

def verify_cleaned_data():
    """
    Verify the cleaned data files
    """
    files_to_check = ['measurements_CAESAR_male_somatotype.csv', 'measurements_CAESAR_female_somatotype.csv']
    
    for filename in files_to_check:
        try:
            df = pd.read_csv(filename)
            print(f"\n{filename}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Missing values per column:")
            missing_counts = df.isnull().sum()
            if missing_counts.sum() == 0:
                print("    No missing values found!")
            else:
                print(missing_counts[missing_counts > 0])
                
        except FileNotFoundError:
            print(f"File {filename} not found.")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Example usage
if __name__ == "__main__":
    # Input file
    input_file = 'CAESAR_Anthro_US.csv'
    
    try:
        # Clean and split data
        male_df, female_df = clean_and_split_anthropometric_data(input_file)
        
        print("\nData processing completed successfully!")
        
        # Verify the results
        print("\n" + "="*60)
        print("VERIFICATION OF CLEANED FILES:")
        print("="*60)
        verify_cleaned_data()
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found. Please check the file path.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")