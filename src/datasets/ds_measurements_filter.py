import pandas as pd
import numpy as np


import sys

sys.path.append("..")
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error


DS_DIR = "data/datasets/"
DS_ANSUR_DIR = os.path.join(DS_DIR, "ds_ansur_original")
DS_SPRING_DIR = os.path.join(DS_DIR, "ds_SPRING")

ANSURI_MEAS = [
    "WEIGHT",
    "STATURE",
    "NECK_CIRC-BASE",
    "CHEST_CIRC",
    "WAIST_CIRC-OMPHALION",  #'WAIST_CIRC_NATURAL',
    "BUTTOCK_CIRC",
    "SHOULDER_CIRC",
    "THIGH_CIRC-PROXIMAL",
    "THIGH_CIRC-DISTAL",
    "CALF_CIRC",
    "ANKLE_CIRC",
    "FOREARM_CIRC-FLEXED",
    "WRIST_CIRC-STYLION",
    "SHOULDER_LNTH",
    "SLEEVE-OUTSEAM_LNTH",
    "RADIALE-STYLION_LNTH",
    "CROTCH_HT",
    "WAIST_NAT_LNTH",
    "THIGH_LINK",  ## needs to be added ((D37) THIGH_LINK = TROCHANTERION_HT - LATERAL_FEMORAL_EPICONDYLE_HT)
    "CHEST_DEPTH",
    "HEAD_CIRC",
    # for somatotype predictors
    "BIACROMIAL_BRTH",
    "SCYE_CIRC_OVER_ACROMION",
]

ANSURII_MEAS = [
    "weightkg",
    "stature",
    "neckcircumferencebase",
    "chestcircumference",
    "waistcircumference",
    "buttockcircumference",
    "shouldercircumference",
    "thighcircumference",
    "lowerthighcircumference",
    "calfcircumference",
    "anklecircumference",
    "forearmcircumferenceflexed",
    "wristcircumference",
    "shoulderlength",
    "sleeveoutseam",
    "radialestylionlength",
    "crotchheight",
    "waistbacklength",
    "thighlink",
    "chestdepth",
    "headcircumference",
    # for somatotype predictors
    "biacromialbreadth",
    "scyecircoveracromion",  # estimated_scye_circ = chestcircumference * (interscyei / chestbreadth)
]

DS_MEAS = {DATASETS[0]: ANSURI_MEAS, DATASETS[1]: ANSURII_MEAS}

NEW_NAMES_DICT_ansurI = {
    ANSURI_MEAS[i]: MEASUREMENTS[i] for i in range(len(ANSURI_MEAS))
}
NEW_NAMES_DICT_ansurII = {
    ANSURII_MEAS[i]: MEASUREMENTS[i] for i in range(len(ANSURII_MEAS))
}

def generateANSURfiles():

    for ds in DATASETS:
        for gender in GENDERS:
            try:
                dataset = load_ds(ds, gender)
                
                # Substitutions, renames and drops
                df = filter_ds(dataset, ds)

                # Save to csv
                file_path = os.path.join(DS_DIR, f"measurements_{ds}_{gender}.csv")
                df.to_csv(file_path, index=False)
                print(f"Generated: {file_path}")
                
            except FileNotFoundError as e:
                print(f"Error: Could not find file for {ds}_{gender}: {e}")
            except Exception as e:
                print(f"Error processing {ds}_{gender}: {e}")
                raise


def load_ds(ds, gender):

    ds_dir = os.path.join(DS_ANSUR_DIR, f"{ds}_{gender}.csv")
    df = pd.read_csv(ds_dir, encoding=FILE_ENCODING, converters={"ID": str})

    return df


def filter_ds(df, ds):

    # Thigh length
    if ds == "ANSURI":
        # (D37) THIGH_LINK = TROCHANTERION_HT - LATERAL_FEMORAL_EPICONDYLE_HT
        df["THIGH_LINK"] = df.apply(
            lambda row: row.TROCHANTERION_HT - row.LATERAL_FEMORAL_EPICONDYLE_HT, axis=1
        )
        # Rename columns and only consider MEASUREMENTS
        df = df.rename(columns=NEW_NAMES_DICT_ansurI, inplace=False)
    elif ds == "ANSURII":
        # estimated_scye_circ = chestcircumference * (interscyei / chestbreadth)
        for c in ("chestcircumference", "interscyei", "chestbreadth"):
            if c not in df.columns:
                df[c] = np.nan

        # ensure numeric
        chest = pd.to_numeric(df["chestcircumference"], errors="coerce")
        inters = pd.to_numeric(df["interscyei"], errors="coerce")
        breadth = pd.to_numeric(df["chestbreadth"], errors="coerce")

        # safe ratio (breadth small -> nan)
        breadth_safe = breadth.replace({0: np.nan})
        ratio = inters / breadth_safe

        scye = chest * ratio

        # sanitize: remove infinities/nans
        scye = scye.replace([np.inf, -np.inf], np.nan)

        # realistic plausible range in RAW UNITS (mm) — adjust if your files use different units
        plausible_min_mm = 300.0   # 30 cm
        plausible_max_mm = 2000.0  # 200 cm
        scye = scye.where((scye >= plausible_min_mm) & (scye <= plausible_max_mm), other=np.nan)

        # Convert to cm (since ANSURII is in mm, and we divide whole df by 10 later)
        scye = scye / 10.0

        df["scyecircoveracromion"] = scye

        # Impute NaNs with median to avoid NaN propagation in training
        if df["scyecircoveracromion"].isna().any():
            median_val = df["scyecircoveracromion"].median()
            df["scyecircoveracromion"] = df["scyecircoveracromion"].fillna(median_val)
            print(f"Imputed {df['scyecircoveracromion'].isna().sum()} NaNs in scyecircoveracromion with median {median_val:.2f}")

        # compute thigh_length vectorized (avoid per-row apply)
        df["thigh_length"] = pd.to_numeric(df.get("trochanterionheight", np.nan), errors="coerce") - pd.to_numeric(
            df.get("lateralfemoralepicondyleheight", np.nan), errors="coerce"
        )
        # rename columns and only consider MEASUREMENTS
        df = df.rename(columns=NEW_NAMES_DICT_ansurII, inplace=False)
    else:
        # d29 = 'trochanterionheight' - 'lateralfemoralepicondyleheight'
        df["thigh_length"] = df.apply(
            lambda row: row.trochanterionheight - row.lateralfemoralepicondyleheight,
            axis=1,
        )
        # Rename columns and only consider MEASUREMENTS
        df = df.rename(columns=NEW_NAMES_DICT_ansurII, inplace=False)

    # Drop columns that are not MEASUREMENTS
    # Divide by 10 to convert to cm (only for ANSURII, ANSURI already in cm)
    # Only use available columns
    available_measurements = [col for col in MEASUREMENTS if col in df.columns]
    df = df[available_measurements]
    if ds == "ANSURII":
        df = df / 10.0

    return df


def generateTOTALfiles():
    for gender in GENDERS:
        try:
            total_data = []  # List to store DataFrames for this gender

            # # Try SPRING (optional)
            # file_path_spring = os.path.join(DS_DIR, f"measurements_SPRING_{gender}.csv")
            # if os.path.exists(file_path_spring):
            #     df_spring = pd.read_csv(file_path_spring, encoding=FILE_ENCODING, converters={"ID": str})
            #     df_spring = df_spring[[c for c in MEASUREMENTS if c in df_spring.columns]]
            #     total_data.append(df_spring)
            #     np.save(os.path.join(DS_DIR, f"measurements_SPRING_{gender}.npy"), df_spring.to_numpy())
            #     print(f"Generated NPY: measurements_SPRING_{gender}.npy")

            # ANSURI
            file_path_ansurI = os.path.join(DS_DIR, f"measurements_ANSURI_{gender}.csv")
            if not os.path.exists(file_path_ansurI):
                raise FileNotFoundError(f"ANSURI file not found: {file_path_ansurI}. Run generateANSURfiles() first.")
            df_ansurI = pd.read_csv(file_path_ansurI, encoding=FILE_ENCODING, converters={"ID": str})
            df_ansurI = df_ansurI[[c for c in MEASUREMENTS if c in df_ansurI.columns]]
            total_data.append(df_ansurI)
            np.save(os.path.join(DS_DIR, f"measurements_ANSURI_{gender}.npy"), df_ansurI.to_numpy())
            print(f"Generated NPY: measurements_ANSURI_{gender}.npy")

            # ANSURII
            file_path_ansurII = os.path.join(DS_DIR, f"measurements_ANSURII_{gender}.csv")
            if not os.path.exists(file_path_ansurII):
                raise FileNotFoundError(f"ANSURII file not found: {file_path_ansurII}. Run generateANSURfiles() first.")
            df_ansurII = pd.read_csv(file_path_ansurII, encoding=FILE_ENCODING, converters={"ID": str})
            df_ansurII = df_ansurII[[c for c in MEASUREMENTS if c in df_ansurII.columns]]
            total_data.append(df_ansurII)
            np.save(os.path.join(DS_DIR, f"measurements_ANSURII_{gender}.npy"), df_ansurII.to_numpy())
            print(f"Generated NPY: measurements_ANSURII_{gender}.npy")

            # Concatenate all DataFrames into a single DataFrame
            total_dataframe = pd.concat(total_data, ignore_index=True)

            # Save the concatenated DataFrame to a new CSV file
            file_path = os.path.join(DS_DIR, f"measurements_TOTAL_{gender}.csv")
            total_dataframe.to_csv(file_path, index=False, encoding=FILE_ENCODING)

            # Save the concatenated DataFrame as npy file
            file_path_npy = os.path.join(DS_DIR, f"measurements_TOTAL_{gender}.npy")
            np.save(file_path_npy, total_dataframe.to_numpy())

            print(f"Generated TOTAL files for {gender}: CSV and NPY (included available datasets)")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise
        except Exception as e:
            print(f"Error processing TOTAL files for {gender}: {e}")
            raise

# ===========================================================================

if __name__ == "__main__":

    generateANSURfiles()

    generateTOTALfiles()
