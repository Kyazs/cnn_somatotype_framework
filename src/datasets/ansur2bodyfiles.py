import os
import pandas as pd
import numpy as np

import sys
sys.path.append("..")
from utils import *

sys.path.append("../reshaper")
from avatar import Avatar


def main():
    '''

    '''

    for ds in DATASETS:

        print(ds)

        ds_dir = os.path.join(OBJ_FILES_DIR, f"obj_database_{ds}")
        try: 
            os.mkdir(ds_dir) 
        except OSError as error: 
            print(error)  
                
        for gender in GENDERS:
            
            file_path = os.path.join(DS_DIR, f"measurements_{ds}_{gender}.csv") #.replace("\\","/")
            
            # Check if the file has bodyID column
            with open(file_path, 'r') as f:
                header = f.readline().strip()
            
            if header.startswith('bodyID'):
                # File has bodyID column, use it as index
                ansur_df = pd.read_csv(file_path, encoding = FILE_ENCODING, index_col=0)
            else:
                # File has no bodyID column, create sequential index
                ansur_df = pd.read_csv(file_path, encoding = FILE_ENCODING)
                ansur_df.index = [f"{ds}_{gender[:3]}_{str(i+1).zfill(4)}" for i in range(len(ansur_df))]
        
            print(ansur_df.shape)
        
            gender_dir = os.path.join(ds_dir, gender)
            try: 
                os.mkdir(gender_dir) 
            except OSError as error: 
                print(error)  
        
            for idx, (index, row) in enumerate(ansur_df.iterrows()):

                ava_name = f"{ds}_{gender[:3]}_{str(idx+1).zfill(4)}"
                ava_path = os.path.join(gender_dir, f"{ava_name}.obj")

                if os.path.exists(ava_path) and check_obj_complete(ava_path):
                    print("File path:", ava_path, "exists")
                else:
                    measurements = list(row)
                    measurements = np.array(measurements).transpose()

                    body = Avatar(measurements, gender)
                    ## Create 3D avatar
                    body.create_obj_file(ava_dir=gender_dir, ava_name=ava_name)


def check_obj_complete(filename, expected_vertices=V_NUM):
    """
    Check if a file has less vertices than expected.

    Parameters:
    - filename (str): The path to the file to check.
    - expected_vertices (int): The expected number of vertices in the file.

    Returns:
    - True if the file is complete (has the expected number of vertices), False otherwise.
    """
    with open(filename, "r") as modelf:
        vertices = 0
        for line in modelf:
            if "v " in line:
                vertices += 1

    if vertices != expected_vertices:
        return False
    else:
        return True

######################################################################
if __name__ == "__main__":
    main()