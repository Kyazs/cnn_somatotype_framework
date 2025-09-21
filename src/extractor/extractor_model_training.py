# Import libraries and modules
import numpy as np
import pandas as pd
import gc  # Garbage Collector - use it like gc.collect()
import joblib
import matplotlib.pyplot as plt

import tensorflow as tf

# Replace buggy check with robust detection + memory growth
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"\nGPU Device(s) found: {gpus}\n")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth configured successfully\n")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}\n")
else:
    print("\nNo GPU detected - training will run on CPU\n")

# Enable mixed precision for better performance and memory efficiency
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"Mixed precision enabled: {policy.name}")
    print(f"Compute dtype: {policy.compute_dtype}")
    print(f"Variable dtype: {policy.variable_dtype}\n")
except Exception as e:
    print(f"Mixed precision setup failed: {e}")
    print("Continuing with default float32 precision\n")

from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv2D,
    Flatten,
    Dropout,
    MaxPooling2D,
    concatenate,
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import psutil
import os
import time
import datetime

import sys

sys.path.append("..")
from utils import *

SIL_FILES_DIR_npy = os.path.join(DS_DIR, f"silhouettes_blender{IMG_SIZE_4NN}_npy")

## Load scaler
if TEST_FILES == True:
    if isinstance(SCALER, StandardScaler):
        TOT_SCALER_DIR = os.path.join(
            MODEL_FILES_DIR, f"scalerStd_{TEST_FILES_NUM}_extractor_test.pkl"
        )
    else:
        TOT_SCALER_DIR = os.path.join(
            MODEL_FILES_DIR, f"scalerMinMax_{TEST_FILES_NUM}_extractor_test.pkl"
        )
else:
    if isinstance(SCALER, StandardScaler):
        TOT_SCALER_DIR = os.path.join(MODEL_FILES_DIR, "scalerStd_extractor.pkl")
    else:
        TOT_SCALER_DIR = os.path.join(MODEL_FILES_DIR, "scalerMinMax_extractor.pkl")


def main():

    # Instantiate a model checkpoint callback
    try:
        os.mkdir(MODEL_FILES_DIR)
    except OSError as error:
        print(error)

    #############################################
    ### Load Databases
    #############################################
    df_total = load_databases()
    print(f"\ndf_total.shape = {df_total.shape}")

    #############################################
    ### Load Images
    #############################################
    # exit()
    print(f"loading images")
    imgX_front, imgX_side = load_images()
    print(f"imgX_front.shape = {imgX_front.shape}")
    print(f"imgX_side.shape = {imgX_side.shape}\n")

    #############################################
    ### Data Alignment - for data sample IDs
    #############################################
    df_total, imgX_front, imgX_side = align_data_by_ids(df_total, imgX_front, imgX_side)

    #############################################
    ### Prepare Numerical and Categorical Data
    #############################################
    
    ## Train and Test splitting
    (trainData, testData, trainImgXf, testImgXf, trainImgXs, testImgXs) = (
        train_test_split(
            df_total,
            imgX_front,
            imgX_side,
            train_size=0.8,
            shuffle=True,
            random_state=123,
        )
    )

    ## DELETE to free memory
    try:
        del imgX_front
    except NameError:
        print("imgX_front was already deleted")

    try:
        del imgX_side
    except NameError:
        print("imgX_side was already deleted")

    gc.collect()

    ## Normalize the UNKNOWN MEASUREMENTS (labels) - will lead to better training and convergence
    trainMeasY = (trainData[UK_MEAS] - df_total[UK_MEAS].mean()) / df_total[
        UK_MEAS
    ].std()
    testMeasY = (testData[UK_MEAS] - df_total[UK_MEAS].mean()) / df_total[UK_MEAS].std()

    ## Scales Train and Test values
    (trainMeasX, testMeasX), dataMeasX_scaler = process_db_values(
        df_total, trainData, testData
    )

    global INP_SHAPE
    INP_SHAPE = trainMeasX.shape[1]

    if "gender" not in KN_MEAS:
        trainMeasX = np.delete(trainMeasX, 0, 1)
        testMeasX = np.delete(testMeasX, 0, 1)

    print(f"\ntrainMeasX.shape = {trainMeasX.shape}")
    print(f"testMeasX.shape = {testMeasX.shape}\n")

    #############################################
    ### Create tf.data Datasets for Memory Efficiency
    #############################################
    
    batch_size = 32  # NN_PARAMETERS["batch_size"]
    
    # Create memory-efficient tf.data datasets
    train_dataset, val_dataset = create_tf_datasets(
        trainMeasX, trainImgXf, trainImgXs, trainMeasY,
        testMeasX, testImgXf, testImgXs, testMeasY,
        batch_size=batch_size
    )

    #############################################
    ### MLP + CNN Model
    #############################################

    ### Data Augmentation
    # https://stackoverflow.com/questions/57092637/how-to-fit-keras-imagedatagenerator-for-large-data-sets-using-batches
    # https://stackoverflow.com/questions/49404993/how-to-use-fit-generator-with-multiple-inputs

    # we create two instances with the same arguments
    data_gen_args = dict(
        featurewise_center=True,
        featurewise_std_normalization=True,
        # rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.25,
        shear_range=3,
        zoom_range=[0.8, 1.2],
        horizontal_flip=False,
    )

    ## compute quantities required for featurewise normalization
    ## (std, mean, and principal components if ZCA whitening is applied)
    datagen = ImageDataGenerator(**data_gen_args)
    datagen.fit(trainImgXf, augment=True)

    ##Create models
    MLP_model = createMLP_model(in_MLPlayers=2)
    print(MLP_model.summary())

    CNN_model = createCNN_model(
        in_CNNlayers=1,
        in_DENSElayers=0,
    )
    # CNN_model = createAlexNetModel()
    print(CNN_model.summary())

    Combined_model = createCombined_model(MLP_model, CNN_model)
    print(Combined_model.summary())

    #############################################
    ### Setup Robust Training Infrastructure
    #############################################
    
    # Create robust callbacks for monitoring and checkpointing
    callbacks_list, emergency_callback = create_robust_callbacks(MODEL_FILES_DIR, TEST_FILES)
    
    # Check for checkpoint resume
    resume_from_checkpoint = emergency_callback.check_for_resume()
    if resume_from_checkpoint:
        try:
            print("Loading model from emergency checkpoint...")
            Combined_model = tf.keras.models.load_model(emergency_callback.emergency_file)
            print("Successfully resumed from checkpoint!")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting training from scratch...")

    #############################################
    ## Compile with Mixed Precision Support
    #############################################
    ## Compile the model using MeanSquaredError as our loss,
    ## implying that we seek to minimize the squared difference - mse
    opt = Adam(1e-4)
    # opt = build_optimizer(NN_PARAMETERS["optimizer"], NN_PARAMETERS["learning_rate"])
    lo = "mse"
    met = ["mean_absolute_error"]
    
    # For mixed precision, we need to wrap the optimizer
    if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
        print("Using loss-scaled optimizer for mixed precision training")
    
    Combined_model.compile(loss=lo, optimizer=opt, metrics=met)

    print(f"\n{'='*60}")
    print("ROBUST TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Mixed Precision: {tf.keras.mixed_precision.global_policy().name}")
    print(f"Optimizer: {type(opt).__name__}")
    print(f"Batch Size: {batch_size}")
    print(f"Dataset Type: tf.data.Dataset (memory optimized)")
    print(f"Callbacks: {len(callbacks_list)} monitoring callbacks enabled")
    print(f"{'='*60}\n")

    """### Train and save the model with robust infrastructure"""

    if TEST_FILES == True:
        EPOCHS = 20
    else:
        EPOCHS = 500

    ## Train the model with robust monitoring
    print("[INFO] Starting robust training with comprehensive monitoring...")
    try:
        Combined_history = Combined_model.fit(
            train_dataset,
            validation_data=val_dataset,
            verbose=2,
            epochs=EPOCHS,
            callbacks=callbacks_list,
        )
        
        print("\n[SUCCESS] Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
        print("Emergency checkpoint should be available for resuming")
        return
        
    except Exception as e:
        print(f"\n[ERROR] Training failed with exception: {e}")
        print("Emergency checkpoint should be available for resuming")
        print("Check system resources and try resuming from checkpoint")
        raise

    gc.collect()

    final_epochs = len(Combined_history.history["val_loss"])

    if TEST_FILES == True:
        if isinstance(SCALER, StandardScaler):
            MODEL_NAME = f"extractor_stdScal_img{IMG_SIZE_4NN}_inp{len(KN_MEAS)}_out{len(UK_MEAS)}_ep{final_epochs}_{TEST_FILES_NUM}test"
        else:
            MODEL_NAME = f"extractor_minMax_img{IMG_SIZE_4NN}_inp{len(KN_MEAS)}_out{len(UK_MEAS)}_ep{final_epochs}_{TEST_FILES_NUM}test"
    else:
        if isinstance(SCALER, StandardScaler):
            MODEL_NAME = f"extractor_stdScal_img{IMG_SIZE_4NN}_inp{len(KN_MEAS)}_out{len(UK_MEAS)}_ep{final_epochs}"
        else:
            MODEL_NAME = f"extractor_minMax_img{IMG_SIZE_4NN}_inp{len(KN_MEAS)}_out{len(UK_MEAS)}_ep{final_epochs}"

    MODEL_NAME_SV = MODEL_NAME + ".keras"
    Combined_model.save(os.path.join(MODEL_FILES_DIR, MODEL_NAME_SV))

    ### Plot history
    histplot(Combined_history, MODEL_NAME_SV, "mean_absolute_error")

def align_data_by_ids(df_total, imgX_front, imgX_side):
    """
    Align image data with measurement data by truncating to minimum sample count.
    This ensures we keep the correct samples that have both measurements and images.
    """
    print("Aligning data by sample count...")
    
    print(f"Measurement data samples: {len(df_total)}")
    print(f"Front image samples: {len(imgX_front)}")
    print(f"Side image samples: {len(imgX_side)}")
    
    # Find minimum sample count
    min_samples = min(len(df_total), len(imgX_front), len(imgX_side))
    print(f"Aligning to minimum sample count: {min_samples}")
    
    # Use the first min_samples that should correspond to the same order
    # This assumes the data was processed in the same order
    df_aligned = df_total.iloc[:min_samples].reset_index(drop=True)
    imgX_front_aligned = imgX_front[:min_samples]
    imgX_side_aligned = imgX_side[:min_samples]
    
    print(f"After alignment:")
    print(f"df_total.shape = {df_aligned.shape}")
    print(f"imgX_front.shape = {imgX_front_aligned.shape}")
    print(f"imgX_side.shape = {imgX_side_aligned.shape}")
    
    return df_aligned, imgX_front_aligned, imgX_side_aligned

def load_databases():
    """
    Loads and prepares the databases for further processing.
    """
    file_encoding = "ISO-8859-1"

    # Initialize lists for each database
    df_ansuri = [None, None]  # [female, male]
    df_ansurii = [None, None]  # [female, male]

    # Explicitly load ANSURI and ANSURII databases
    target_databases = ["ANSURI", "ANSURII"]
    
    for db_name in target_databases:
        for i, gender in enumerate(GENDER_DICT.keys()):
            db_dir = os.path.join(DS_DIR, f"measurements_{db_name}_{gender}.csv")
            
            # Check if the file exists
            if not os.path.exists(db_dir):
                print(f"ERROR: File {db_dir} does not exist!")
                print("Available files:")
                import glob
                available_files = glob.glob(os.path.join(DS_DIR, "measurements_*.csv"))
                for f in available_files:
                    print(f"  - {os.path.basename(f)}")
                sys.exit(1)
                
            print(f"Loading: {db_dir}")
            df = pd.read_csv(db_dir, encoding=file_encoding, converters={"ID": str})
            
            # Remove unwanted columns
            df.drop(
                labels=df.columns.difference(UK_MEAS + KN_MEAS),
                axis=1,
                inplace=True,
            )
            df["gender"] = i
            
            # Assign to appropriate list
            if db_name == "ANSURI":
                df_ansuri[i] = df
            elif db_name == "ANSURII":
                df_ansurii[i] = df

    # Validate that all required dataframes were loaded
    if df_ansuri[0] is None or df_ansuri[1] is None:
        print("ERROR: Missing ANSURI database files")
        sys.exit(1)
        
    if df_ansurii[0] is None or df_ansurii[1] is None:
        print("ERROR: Missing ANSURII database files")
        sys.exit(1)

    print(f"Loaded ANSURI: {len(df_ansuri[0])} female, {len(df_ansuri[1])} male")
    print(f"Loaded ANSURII: {len(df_ansurii[0])} female, {len(df_ansurii[1])} male")

    # Concatenate DataFrames
    df_female = pd.concat([df_ansuri[0], df_ansurii[0]], axis=0)
    df_male = pd.concat([df_ansuri[1], df_ansurii[1]], axis=0)

    if TEST_FILES == True:
        df_female = df_female.head(2 * TEST_FILES_NUM)  # 2 datasets
        df_male = df_male.head(2 * TEST_FILES_NUM)

    df_total = pd.concat([df_female, df_male], axis=0)
    
    print(f"Total combined dataset: {len(df_total)} samples")
    return df_total

def load_images():
    """
    Loads images from .npz files and concatenates them.
    Images are loaded from ANSURI2023 and ANSURII2023 datasets for both genders.

    Returns:
        Tuple of numpy arrays :
            - imgX_front (numpy array) : concatenated front view images.
            - imgX_side (numpy array) : concatenated side view images.
    """

    img_ansuri = [[] for x in range(2)]  # 0-female, 1-male
    img_ansurii = [[] for x in range(2)]  # 0-female, 1-male

    total_samples = 0

    for g, gender in enumerate(GENDER_DICT.keys()):
        ## ANSURI
        if TEST_FILES == True:
            npz_file_name = (
                f"silh_Xarray{IMG_SIZE_4NN}_ANSURI_{gender}_bw_{TEST_FILES_NUM}test.npz"
            )
        else:
            npz_file_name = f"silh_Xarray{IMG_SIZE_4NN}_ANSURI_{gender}_bw.npz"

        npz_path = os.path.join(
            SIL_FILES_DIR_npy, f"silhouettes_ANSURI_bw", npz_file_name
        )

        if not os.path.exists(npz_path):
            print(f"ERROR: NPZ file {npz_path} does not exist!")
            sys.exit(1)

        print(f"Loading ANSURI {gender}: {npz_path}")
        img_ansuri_npz = np.load(npz_path, allow_pickle=True)
        print(f"  - Shape: {img_ansuri_npz['arr_0'].shape}")
        print(f"  - Samples: {img_ansuri_npz['arr_0'].shape[1]}")
        total_samples += img_ansuri_npz['arr_0'].shape[1]

        ## ANSURII
        if TEST_FILES == True:
            npz_file_name = f"silh_Xarray{IMG_SIZE_4NN}_ANSURII_{gender}_bw_{TEST_FILES_NUM}test.npz"
        else:
            npz_file_name = f"silh_Xarray{IMG_SIZE_4NN}_ANSURII_{gender}_bw.npz"

        npz_path = os.path.join(
            SIL_FILES_DIR_npy, f"silhouettes_ANSURII_bw", npz_file_name
        )

        if not os.path.exists(npz_path):
            print(f"ERROR: NPZ file {npz_path} does not exist!")
            sys.exit(1)

        print(f"Loading ANSURII {gender}: {npz_path}")
        img_ansurii_npz = np.load(npz_path, allow_pickle=True)
        print(f"  - Shape: {img_ansurii_npz['arr_0'].shape}")
        print(f"  - Samples: {img_ansurii_npz['arr_0'].shape[1]}")
        total_samples += img_ansurii_npz['arr_0'].shape[1]

        for i, view in enumerate(VIEWS):
            img_ansuri[g].append(img_ansuri_npz["arr_0"][i, :, :, :])
            img_ansurii[g].append(img_ansurii_npz["arr_0"][i, :, :, :])

    print(f"Total image samples loaded: {total_samples}")

    # ... rest of the function remains the same ...
    
    ## DELETE to free memory
    try:
        del img_ansuri_npz
    except NameError:
        print("img_ansuri_npz was already deleted")

    try:
        del img_ansurii_npz
    except NameError:
        print("img_ansurii_npz was already deleted")

    gc.collect()

    ## Concatenate NPYarrays
    imgX_front_female = np.concatenate((img_ansuri[0][0], img_ansurii[0][0]), axis=0)
    imgX_front_male = np.concatenate((img_ansuri[1][0], img_ansurii[1][0]), axis=0)

    imgX_side_female = np.concatenate((img_ansuri[0][1], img_ansurii[0][1]), axis=0)
    imgX_side_male = np.concatenate((img_ansuri[1][1], img_ansurii[1][1]), axis=0)

    print(f"After concatenation:")
    print(f"  - Front female: {imgX_front_female.shape}")
    print(f"  - Front male: {imgX_front_male.shape}")
    print(f"  - Side female: {imgX_side_female.shape}") 
    print(f"  - Side male: {imgX_side_male.shape}")

    ## DELETE to free memory
    try:
        del img_ansuri
    except NameError:
        print("img_ansuri was already deleted")

    try:
        del img_ansurii
    except NameError:
        print("img_ansurii was already deleted")

    gc.collect()

    imgX_front = np.concatenate((imgX_front_female, imgX_front_male), axis=0)
    imgX_side = np.concatenate((imgX_side_female, imgX_side_male), axis=0)

    print(f"Final shapes:")
    print(f"  - imgX_front: {imgX_front.shape}")
    print(f"  - imgX_side: {imgX_side.shape}")

    ## DELETE to free memory
    try:
        del imgX_front_female
    except NameError:
        print("imgX_front_female was already deleted")

    try:
        del imgX_front_male
    except NameError:
        print("imgX_front_male was already deleted")

    try:
        del imgX_side_female
    except NameError:
        print("imgX_side_female was already deleted")

    try:
        del imgX_side_male
    except NameError:
        print("imgX_side_male was already deleted")

    return imgX_front, imgX_side


def create_tf_datasets(trainMeasX, trainImgXf, trainImgXs, trainMeasY, 
                      testMeasX, testImgXf, testImgXs, testMeasY, batch_size=32):
    """
    Create tf.data.Dataset objects for better memory management and performance.
    
    Args:
        trainMeasX, trainImgXf, trainImgXs, trainMeasY: Training data
        testMeasX, testImgXf, testImgXs, testMeasY: Testing data
        batch_size: Batch size for training
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    print("Creating tf.data datasets for memory-efficient training...")
    
    # Convert to float32 for consistency and mixed precision compatibility
    trainMeasX = tf.cast(trainMeasX, tf.float32)
    trainImgXf = tf.cast(trainImgXf, tf.float32) 
    trainImgXs = tf.cast(trainImgXs, tf.float32)
    trainMeasY = tf.cast(trainMeasY, tf.float32)
    
    testMeasX = tf.cast(testMeasX, tf.float32)
    testImgXf = tf.cast(testImgXf, tf.float32)
    testImgXs = tf.cast(testImgXs, tf.float32) 
    testMeasY = tf.cast(testMeasY, tf.float32)
    
    print(f"Data shapes after conversion:")
    print(f"  trainMeasX: {trainMeasX.shape}")
    print(f"  trainImgXf: {trainImgXf.shape}")
    print(f"  trainImgXs: {trainImgXs.shape}")
    print(f"  trainMeasY: {trainMeasY.shape}")
    
    # Create separate datasets for each input to avoid shape conflicts
    train_num_dataset = tf.data.Dataset.from_tensor_slices(trainMeasX)
    train_front_dataset = tf.data.Dataset.from_tensor_slices(trainImgXf)
    train_side_dataset = tf.data.Dataset.from_tensor_slices(trainImgXs)
    train_target_dataset = tf.data.Dataset.from_tensor_slices(trainMeasY)
    
    val_num_dataset = tf.data.Dataset.from_tensor_slices(testMeasX)
    val_front_dataset = tf.data.Dataset.from_tensor_slices(testImgXf)
    val_side_dataset = tf.data.Dataset.from_tensor_slices(testImgXs)
    val_target_dataset = tf.data.Dataset.from_tensor_slices(testMeasY)
    
    # Zip datasets together (this maintains separate tensors)
    train_dataset = tf.data.Dataset.zip((train_num_dataset, train_front_dataset, train_side_dataset, train_target_dataset))
    val_dataset = tf.data.Dataset.zip((val_num_dataset, val_front_dataset, val_side_dataset, val_target_dataset))
    
    # Configure training dataset with optimizations
    def format_inputs(numerical, front_img, side_img, targets):
        """Format inputs for the model: returns ([num, front, side], targets)"""
        return [numerical, front_img, side_img], targets
    
    train_dataset = (train_dataset
                    .shuffle(buffer_size=1000)
                    .batch(batch_size)
                    .map(format_inputs, num_parallel_calls=tf.data.AUTOTUNE)
                    .prefetch(tf.data.AUTOTUNE))
    
    # Configure validation dataset
    val_dataset = (val_dataset
                  .batch(batch_size)
                  .map(format_inputs, num_parallel_calls=tf.data.AUTOTUNE)
                  .prefetch(tf.data.AUTOTUNE))
    
    print(f"Train dataset created successfully")
    print(f"Validation dataset created successfully")
    
    return train_dataset, val_dataset


class MemoryMonitorCallback(tf.keras.callbacks.Callback):
    """Monitor system memory and GPU memory usage during training."""
    
    def __init__(self, log_frequency=10):
        super().__init__()
        self.log_frequency = log_frequency
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_frequency == 0:
            # System memory
            memory = psutil.virtual_memory()
            print(f"\n[Memory Monitor] Epoch {epoch}:")
            print(f"  System RAM: {memory.percent:.1f}% used ({memory.used/1e9:.1f}GB/{memory.total/1e9:.1f}GB)")
            
            # GPU memory if available
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                try:
                    gpu_details = tf.config.experimental.get_memory_info('GPU:0')
                    gpu_used = gpu_details['current'] / 1e6  # Convert to MB
                    gpu_peak = gpu_details['peak'] / 1e6
                    print(f"  GPU Memory: Current {gpu_used:.0f}MB, Peak {gpu_peak:.0f}MB")
                except:
                    print("  GPU memory info not available")
            
            # Check for memory pressure
            if memory.percent > 90:
                print(f"  WARNING: High system memory usage ({memory.percent:.1f}%)")
            
            print()


class EmergencyCheckpointCallback(tf.keras.callbacks.Callback):
    """Save emergency checkpoints and handle recovery from crashes."""
    
    def __init__(self, checkpoint_dir, save_frequency=25):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.save_frequency = save_frequency
        self.emergency_file = os.path.join(checkpoint_dir, "emergency_checkpoint.keras")
        self.training_log = os.path.join(checkpoint_dir, "training_progress.txt")
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize training log
        with open(self.training_log, 'w') as f:
            f.write(f"Training started at: {datetime.datetime.now()}\n")
    
    def on_train_begin(self, logs=None):
        print(f"Emergency checkpoints will be saved to: {self.checkpoint_dir}")
        print(f"Checkpoint frequency: every {self.save_frequency} epochs")
    
    def on_epoch_end(self, epoch, logs=None):
        # Save emergency checkpoint periodically
        if epoch % self.save_frequency == 0:
            self.model.save(self.emergency_file, overwrite=True)
            print(f"\n[Emergency Checkpoint] Saved at epoch {epoch}")
        
        # Log progress
        with open(self.training_log, 'a') as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            loss = logs.get('loss', 'N/A')
            val_loss = logs.get('val_loss', 'N/A')
            f.write(f"{timestamp} - Epoch {epoch}: loss={loss}, val_loss={val_loss}\n")
    
    def check_for_resume(self):
        """Check if there's a checkpoint to resume from."""
        if os.path.exists(self.emergency_file):
            print(f"\nFound emergency checkpoint: {self.emergency_file}")
            response = input("Do you want to resume from this checkpoint? (y/n): ")
            if response.lower() == 'y':
                return True
        return False


class TrainingMonitorCallback(tf.keras.callbacks.Callback):
    """Monitor training progress and detect anomalies."""
    
    def __init__(self):
        super().__init__()
        self.loss_history = []
        self.val_loss_history = []
        self.training_start_time = None
        self.epoch_times = []
    
    def on_train_begin(self, logs=None):
        self.training_start_time = time.time()
        print(f"\n[Training Monitor] Training started at {datetime.datetime.now()}")
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        
        loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        self.loss_history.append(loss)
        self.val_loss_history.append(val_loss)
        
        # Check for training anomalies
        if epoch > 10:  # Wait for some epochs before checking
            recent_avg = np.mean(self.loss_history[-5:])
            overall_avg = np.mean(self.loss_history)
            
            # Check for training stagnation
            if len(self.loss_history) > 20:
                recent_20 = self.loss_history[-20:]
                if max(recent_20) - min(recent_20) < 0.001:  # Very small variation
                    print(f"\n[Warning] Training may be stagnating (loss variation < 0.001 over last 20 epochs)")
            
            # Check for unusual epoch times
            if len(self.epoch_times) > 5:
                avg_time = np.mean(self.epoch_times[-10:])
                if epoch_time > avg_time * 2:
                    print(f"\n[Warning] Epoch {epoch} took unusually long: {epoch_time:.1f}s (avg: {avg_time:.1f}s)")
        
        # Estimate remaining time
        if epoch > 0:
            avg_epoch_time = np.mean(self.epoch_times)
            total_epochs = self.params.get('epochs', 500)
            remaining_epochs = total_epochs - epoch - 1
            remaining_time = remaining_epochs * avg_epoch_time
            hours, remainder = divmod(remaining_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            if epoch % 50 == 0:  # Print every 50 epochs
                print(f"\n[Training Monitor] Estimated remaining time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    
    def on_train_end(self, logs=None):
        total_time = time.time() - self.training_start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\n[Training Monitor] Training completed in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")


def create_robust_callbacks(model_files_dir, test_files=False):
    """Create a comprehensive set of callbacks for robust training."""
    callbacks = []
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss", 
        min_delta=1e-4, 
        patience=15,  # Increased patience for more stability
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Regular model checkpointing (best model)
    checkpoint_file = os.path.join(model_files_dir, "best_model_checkpoint.keras")
    model_checkpoint = ModelCheckpoint(
        checkpoint_file,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    callbacks.append(model_checkpoint)
    
    # Emergency checkpointing
    emergency_dir = os.path.join(model_files_dir, "emergency_checkpoints")
    emergency_callback = EmergencyCheckpointCallback(emergency_dir, save_frequency=25)
    callbacks.append(emergency_callback)
    
    # Memory monitoring
    memory_callback = MemoryMonitorCallback(log_frequency=20)
    callbacks.append(memory_callback)
    
    # Training monitoring
    training_monitor = TrainingMonitorCallback()
    callbacks.append(training_monitor)
    
    # Reduce learning rate on plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    return callbacks, emergency_callback

def process_db_values(df, train, test):
    """
    Performs min-max scaling each continuous feature column to the range [0, 1]

    Parameters:
        df (Pandas.DataFrame): Dataframe containing all features.
        train (Pandas.DataFrame): Dataframe containing the training subset of the data.
        test (Pandas.DataFrame): Dataframe containing the testing subset of the data.

    Returns:
        tuple: Tuple containing the processed training and testing data, and the MinMaxScaler (or StandardScaler) object.
    """

    cs = SCALER
    trainContinuous = cs.fit_transform(train[CONTINUOUS])
    testContinuous = cs.transform(test[CONTINUOUS])  # test[KN_MEAS - CATEGORICAL]

    # one-hot encode the GENDER categorical data (by definition of
    # one-hot encoding, all output features are now in the range [0, 1])
    trainCategorical = keras.utils.to_categorical(
        train[CATEGORICAL], len(GENDER_DICT.keys())
    )
    testCategorical = keras.utils.to_categorical(
        test[CATEGORICAL], len(GENDER_DICT.keys())
    )

    # construct our training and testing data points by concatenating
    # the categorical features with the continuous features
    trainX = np.hstack([trainCategorical, trainContinuous])
    testX = np.hstack([testCategorical, testContinuous])

    # Save the scaler
    joblib.dump(cs, TOT_SCALER_DIR)

    # return the concatenated training and testing data
    return (trainX, testX), cs


## MLP model
## Define the MLP_model
def createMLP_model(in_MLPlayers=1):
    """
    Creates an MLP model for the categorical and numerical inputs,
    which will be used for predicting the target variable

    Args:
        in_MLPlayers: Number of inner MLP layers to use. Default is 1.

    Returns:
        A Keras Model object.
    """

    mlp_input = Input(shape=(INP_SHAPE,), name="mlp_input")

    mlp_hidden = Dense(16, activation="relu", name="mlp_hidden1")(mlp_input)

    for i in range(in_MLPlayers):
        mlp_hidden = Dense(
            64,
            activation="relu",
            name=f"mlp_hiddenInner{i+1}",
        )(mlp_hidden)

    mlp_hidden = Dense(64, activation="relu", name="mlp_hidden2")(mlp_hidden)

    # For mixed precision, ensure output layer uses float32
    if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
        mlp_output = Dense(len(UK_MEAS), activation="linear", name="mlp_output", dtype='float32')(mlp_hidden)
    else:
        mlp_output = Dense(len(UK_MEAS), activation="linear", name="mlp_output")(mlp_hidden)

    ##returns model
    return Model(mlp_input, mlp_output)


KERNEL_SIZE = (3, 3)  #  (3,3)
POOL_SIZE = (3, 3)  #  (3,3)


def createCNN_model(in_CNNlayers=1, in_DENSElayers=0):
    """
    Creates the CNN model with (modified) AlexNet architecture for image inputs

    Args:
        in_CNNlayers : number of hidden convolutional layers
        in_DENSElayers : number of hidden dense layers

    Returns:
        A CNN model with AlexNet architecture
    """
    ## AlexNet-wise ##
    ## MLP-CNN_sil2meas_224_compcan-v2 / sweep_jesus_cnn_compcan / run_jesus_10 ##
    cnn_input = Input(shape=(IMG_SIZE_4NN, IMG_SIZE_4NN, CHAN), name="cnn_input")
    ######################################################################
    ## CNN1
    cnn_hidden = Conv2D(96, (5, 5), activation="relu", name="cnn_hidden1")(cnn_input)
    maxpool = MaxPooling2D(pool_size=POOL_SIZE)(cnn_hidden)

    cnn_hidden = maxpool

    ## CNN inner
    for i in range(in_CNNlayers):
        cnn_hidden = Conv2D(
            128,
            KERNEL_SIZE,
            activation="relu",
            name=f"cnn_hiddenInner{i+1}",
        )(cnn_hidden)

        ## Inner maxpool
        cnn_hidden = MaxPooling2D(pool_size=POOL_SIZE)(cnn_hidden)

    ## CNN2
    cnn_hidden = Conv2D(
        64,
        KERNEL_SIZE,
        activation="relu",
        name="cnn_hidden2",
    )(cnn_hidden)
    maxpool = MaxPooling2D(pool_size=POOL_SIZE)(cnn_hidden)

    flatten = Flatten()(maxpool)

    dense_hidden = Dense(500, activation="relu", name="dense_hidden1")(flatten)
    dense_hidden = Dropout(0.3)(dense_hidden)

    for i in range(in_DENSElayers):
        dense_hidden = Dense(
            200,
            activation="relu",
            name=f"dense_hiddenInner{i+1}",
        )(dense_hidden)
        dense_hidden = Dropout(0.0)(dense_hidden)

    dense_hidden = Dense(200, activation="relu", name="dense_hidden2")(dense_hidden)
    dense_hidden = Dropout(0.5)(dense_hidden)

    ######################################################################

    # For mixed precision, ensure output layer uses float32
    if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
        cnn_output = Dense(len(UK_MEAS), activation="linear", name="cnn_output", dtype='float32')(
            dense_hidden
        )
    else:
        cnn_output = Dense(len(UK_MEAS), activation="linear", name="cnn_output")(
            dense_hidden
        )

    ##returns model
    return Model(cnn_input, cnn_output)


## Here is the function that merges our two generators
## We use the exact same generator with the same random seed for both the front and side images
def generator2imgsNumData(datagen, MeasX, ImgXf, ImgXs, MeasY, batch_size):  # MeasY,
    """
    Generates data for training the model by combining numerical input, front view images, and side view images.

    Args:
        datagen (ImageDataGenerator): An instance of ImageDataGenerator for data augmentation.
        MeasX (array-like): Numerical input data.
        ImgXf (array-like): Front view image data.
        ImgXs (array-like): Side view image data.
        MeasY (array-like): Target variable data.
        batch_size (int): Number of samples to yield per batch.

    Yields:
        tuple: A tuple containing input data [numerical input, front view images, side view images] and the target variable data.
    """

    genNum = datagen.flow(ImgXf, MeasX, batch_size=batch_size, shuffle=False, seed=123)
    genF = datagen.flow(ImgXf, MeasY, batch_size=batch_size, shuffle=False, seed=123)
    genS = datagen.flow(ImgXs, MeasY, batch_size=batch_size, shuffle=False, seed=321)

    while True:
        Xni = next(genNum)
        Xfi = next(genF)
        Xsi = next(genS)
        yield [Xni[1], Xfi[0], Xsi[0]], Xfi[1]


def createCombined_model(MLP_model, CNN_model):
    """
    Creates the final combined model by merging the outputs of the MLP model and the CNN model
    """

    input_numca = Input(shape=(INP_SHAPE,), name="input_numca")
    input_front = Input((IMG_SIZE_4NN, IMG_SIZE_4NN, CHAN), name="input_front")
    input_side = Input((IMG_SIZE_4NN, IMG_SIZE_4NN, CHAN), name="input_side")

    output_numca = MLP_model(input_numca)
    output_front = CNN_model(input_front)
    output_side = CNN_model(input_side)

    ## Create the input to our final set of layers as the *output* of the MLP nad both CNNs
    combinedInput = concatenate(
        [output_numca, output_front, output_side], name="combined_input"
    )

    ## Our final FC layer head will have X dense layers, the final one being our regression head
    combined_hidden = Dense(
        len(UK_MEAS) * 2, activation="relu", name="combined_hidden"
    )(combinedInput)

    # For mixed precision, ensure output layer uses float32
    if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
        combinedOutput = Dense(len(UK_MEAS), activation="linear", name="combined_output", dtype='float32')(
            combined_hidden
        )
    else:
        combinedOutput = Dense(len(UK_MEAS), activation="linear", name="combined_output")(
            combined_hidden
        )

    return Model(inputs=[input_numca, input_front, input_side], outputs=combinedOutput)


def histplot(history, model_name, acc_metric="mean_absolute_error"):
    """
    Plots training and validation loss and accuracy for a model.
    Args:
        history: A Keras training history object.
        model_name: The name of the model being plotted.
        acc_metric: The accuracy metric to be plotted. Defaults to 'mean_absolute_error'.
    """

    hist = pd.DataFrame(history.history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    if TEST_FILES == True:
        fig.suptitle(
            f"Model accuracy - ANSUR_imgInputs DataAugmentation (test{TEST_FILES_NUM}))"
        )
    else:
        fig.suptitle(f"Model accuracy - ANSUR_imgInputs DataAugmentation")

    hist.plot(y=["loss", "val_loss"], ax=ax1)

    min_loss = hist["loss"].min()
    ax1.hlines(
        min_loss,
        0,
        len(hist),
        linestyle="dashed",
        label="min(loss) = {:.3f}".format(min_loss),
    )

    min_val_loss = hist["val_loss"].min()
    ax1.hlines(
        min_val_loss,
        0,
        len(hist),
        linestyle="dotted",
        label="min(val_loss) = {:.3f}".format(min_val_loss),
    )

    ax1.legend(loc="upper right")
    ax1.legend(
        [
            "loss - mse",
            "val_loss - mse",
            f"min(loss - mse) = {round(min_loss, 3)}",
            f"min(val_loss - mse) = {round(min_val_loss, 3)}",
        ]
    )

    hist.plot(y=[f"{acc_metric}", f"val_{acc_metric}"], ax=ax2)

    min_acc = hist[f"{acc_metric}"].min()
    ax2.hlines(
        min_acc,
        0,
        len(hist),
        linestyle="dashed",
        label=f"min({acc_metric})" + " = {:.3f}".format(min_acc),
    )

    min_val_acc = hist[f"val_{acc_metric}"].min()
    ax2.hlines(
        min_val_acc,
        0,
        len(hist),
        linestyle="dotted",
        label=f"min(val_{acc_metric})" + " = {:.3f}".format(min_val_acc),
    )

    ax2.legend(loc="upper right")

    fig_name = model_name.split(".")[0] + "_hist"
    fig.savefig(os.path.join(MODEL_FILES_DIR, f"{fig_name}.png"))


if __name__ == "__main__":

    main()
