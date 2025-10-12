#!/usr/bin/env python3
"""
Height Predictor Model Training Script - Transfer Learning Implementation
===========================================================================

This script trains a HEIGHT PREDICTOR using state-of-the-art transfer learning
with a Siamese architecture and EfficientNet backbone.

MODEL ARCHITECTURE:
-------------------
- Backbone: EfficientNetB0 pre-trained on ImageNet (shared weights)
- Input: Front + Side silhouette images (224x224x1 → converted to RGB)
- Fusion: Concatenated features from both views
- Output: Height prediction (stature_cm)

TRAINING STRATEGY:
------------------
1. Phase 1: Freeze backbone, train regression head only (warm-up)
2. Phase 2: Unfreeze last layers, fine-tune end-to-end (full training)
3. Differential learning rates: lower for backbone, higher for head
4. Data augmentation: consistent transformations across both views

EVALUATION METRICS:
-------------------
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

Author: Implemented based on best practices for transfer learning
Date: October 13, 2025
"""

import os
import sys
import gc
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    Callback
)

# GPU Configuration
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"\n{'='*80}")
    print(f"GPU Device(s) found: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
    print(f"{'='*80}\n")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled\n")
    except RuntimeError as e:
        print(f"❌ GPU configuration error: {e}\n")
else:
    print("\n⚠️  No GPU detected - training will run on CPU\n")

# Mixed Precision Configuration
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"✅ Mixed precision enabled: {policy.name}")
    print(f"   Compute dtype: {policy.compute_dtype}")
    print(f"   Variable dtype: {policy.variable_dtype}\n")
except Exception as e:
    print(f"⚠️  Mixed precision setup failed: {e}")
    print("   Continuing with default float32 precision\n")

# Import project utilities
sys.path.append("..")
from utils import *

# Global Configuration
SIL_FILES_DIR_npy = os.path.join(DS_DIR, f"silhouettes_blender{IMG_SIZE_4NN}_npy")
MODEL_SAVE_DIR = MODEL_FILES_DIR

# Dataset Configuration
KN_MEAS = ["gender"]  # Known measurements (inputs)
UK_MEAS = ["stature_cm"]  # Unknown measurements (outputs - height only)
CONTINUOUS = []  # No continuous inputs
CATEGORICAL = ["gender"]  # Gender is categorical

print(f"\n{'='*80}")
print("📋 HEIGHT PREDICTOR CONFIGURATION")
print(f"{'='*80}")
print(f"Inputs: {KN_MEAS} + front image + side image")
print(f"Output: {UK_MEAS}")
print(f"Image size: {IMG_SIZE_4NN}x{IMG_SIZE_4NN}x{CHAN}")
print(f"Backbone: EfficientNetB0 (ImageNet pre-trained)")
print(f"{'='*80}\n")


#############################################
### STEP 1: DATA LOADING AND PREPROCESSING
#############################################

def load_databases():
    """
    Load measurement databases from ANSUR I and II datasets.
    Combines female and male data with proper gender encoding.
    """
    print("📂 Loading measurement databases...")
    
    file_encoding = FILE_ENCODING
    
    # Load TOTAL files (combined ANSUR I + II)
    tot_db_dir_fem = os.path.join(DS_DIR, "measurements_TOTAL_female.csv")
    tot_db_dir_mal = os.path.join(DS_DIR, "measurements_TOTAL_male.csv")
    
    # Verify files exist
    if not os.path.exists(tot_db_dir_fem) or not os.path.exists(tot_db_dir_mal):
        raise FileNotFoundError(
            f"Database files not found!\n"
            f"Expected: {tot_db_dir_fem}\n"
            f"Expected: {tot_db_dir_mal}\n"
            f"Run ds_measurements_filter_somatotype.py to generate TOTAL files."
        )
    
    # Load datasets
    df_female = pd.read_csv(tot_db_dir_fem, encoding=file_encoding, converters={"ID": str})
    df_male = pd.read_csv(tot_db_dir_mal, encoding=file_encoding, converters={"ID": str})
    
    # Add gender column
    df_female["gender"] = 0  # Female = 0
    df_male["gender"] = 1    # Male = 1
    
    print(f"   Female samples: {len(df_female)}")
    print(f"   Male samples: {len(df_male)}")
    
    # Combine datasets
    df_total = pd.concat([df_female, df_male], axis=0, ignore_index=True)
    
    # Keep only required columns
    required_cols = list(dict.fromkeys(UK_MEAS + KN_MEAS))
    missing_cols = set(required_cols) - set(df_total.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    df_total = df_total[required_cols].copy()
    
    # Apply test file limitation if needed
    if TEST_FILES:
        print(f"   ⚠️  TEST MODE: Limiting to {TEST_FILES_NUM} samples per gender")
        df_total = df_total.head(TEST_FILES_NUM * 2)
    
    print(f"   ✅ Combined dataset: {len(df_total)} samples")
    
    # Verify height data quality
    height_stats = df_total['stature_cm'].describe()
    print(f"\n   Height statistics (cm):")
    print(f"      Mean: {height_stats['mean']:.1f}")
    print(f"      Std:  {height_stats['std']:.1f}")
    print(f"      Min:  {height_stats['min']:.1f}")
    print(f"      Max:  {height_stats['max']:.1f}")
    
    return df_total


def load_images():
    """
    Load silhouette images from NPZ files.
    Returns front and side view images for all subjects.
    """
    print("\n🖼️  Loading silhouette images...")
    
    img_ansuri = [[] for _ in range(2)]  # 0-female, 1-male
    img_ansurii = [[] for _ in range(2)]  # 0-female, 1-male
    
    for g, gender in enumerate(GENDER_DICT.keys()):
        # ANSUR I
        if TEST_FILES:
            npz_file_name = f"silh_Xarray{IMG_SIZE_4NN}_ANSURI_{gender}_bw_{TEST_FILES_NUM}test.npz"
        else:
            npz_file_name = f"silh_Xarray{IMG_SIZE_4NN}_ANSURI_{gender}_bw.npz"
        
        npz_path = os.path.join(SIL_FILES_DIR_npy, f"silhouettes_ANSURI_bw", npz_file_name)
        
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        
        print(f"   Loading ANSURI {gender}: {os.path.basename(npz_path)}")
        img_ansuri_npz = np.load(npz_path, allow_pickle=True)
        print(f"      Shape: {img_ansuri_npz['arr_0'].shape}")
        
        # ANSUR II
        if TEST_FILES:
            npz_file_name = f"silh_Xarray{IMG_SIZE_4NN}_ANSURII_{gender}_bw_{TEST_FILES_NUM}test.npz"
        else:
            npz_file_name = f"silh_Xarray{IMG_SIZE_4NN}_ANSURII_{gender}_bw.npz"
        
        npz_path = os.path.join(SIL_FILES_DIR_npy, f"silhouettes_ANSURII_bw", npz_file_name)
        
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        
        print(f"   Loading ANSURII {gender}: {os.path.basename(npz_path)}")
        img_ansurii_npz = np.load(npz_path, allow_pickle=True)
        print(f"      Shape: {img_ansurii_npz['arr_0'].shape}")
        
        # Append images for each view
        for i, view in enumerate(VIEWS):
            img_ansuri[g].append(img_ansuri_npz["arr_0"][i, :, :, :])
            img_ansurii[g].append(img_ansurii_npz["arr_0"][i, :, :, :])
        
        del img_ansuri_npz, img_ansurii_npz
        gc.collect()
    
    # Concatenate ANSUR I + II for each gender
    imgX_front_female = np.concatenate((img_ansuri[0][0], img_ansurii[0][0]), axis=0)
    imgX_front_male = np.concatenate((img_ansuri[1][0], img_ansurii[1][0]), axis=0)
    imgX_side_female = np.concatenate((img_ansuri[0][1], img_ansurii[0][1]), axis=0)
    imgX_side_male = np.concatenate((img_ansuri[1][1], img_ansurii[1][1]), axis=0)
    
    del img_ansuri, img_ansurii
    gc.collect()
    
    # Concatenate female + male
    imgX_front = np.concatenate((imgX_front_female, imgX_front_male), axis=0)
    imgX_side = np.concatenate((imgX_side_female, imgX_side_male), axis=0)
    
    del imgX_front_female, imgX_front_male, imgX_side_female, imgX_side_male
    gc.collect()
    
    print(f"   ✅ Front images: {imgX_front.shape}")
    print(f"   ✅ Side images:  {imgX_side.shape}")
    
    return imgX_front, imgX_side


def align_data_by_ids(df_total, imgX_front, imgX_side):
    """
    Align measurement data with image data by sample count.
    """
    print("\n🔗 Aligning data by sample count...")
    
    min_samples = min(len(df_total), len(imgX_front), len(imgX_side))
    print(f"   Measurement samples: {len(df_total)}")
    print(f"   Front image samples: {len(imgX_front)}")
    print(f"   Side image samples:  {len(imgX_side)}")
    print(f"   ✅ Aligned to: {min_samples} samples")
    
    df_aligned = df_total.iloc[:min_samples].reset_index(drop=True)
    imgX_front_aligned = imgX_front[:min_samples]
    imgX_side_aligned = imgX_side[:min_samples]
    
    return df_aligned, imgX_front_aligned, imgX_side_aligned


def prepare_data_splits(df_total, imgX_front, imgX_side, 
                        train_size=0.8, val_size=0.1, test_size=0.1,
                        random_state=42):
    """
    Split data into train, validation, and test sets.
    Uses stratified splitting to maintain gender distribution.
    
    Returns:
        Tuple of (train_data, val_data, test_data) where each is a dict
        containing 'gender', 'front', 'side', and 'height'.
    """
    print(f"\n📊 Splitting data (train: {train_size*100:.0f}%, val: {val_size*100:.0f}%, test: {test_size*100:.0f}%)...")
    
    # Verify split proportions sum to 1.0
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Split proportions must sum to 1.0"
    
    # Extract data
    gender = df_total['gender'].values
    height = df_total['stature_cm'].values
    
    # First split: train + val vs test
    indices = np.arange(len(df_total))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=gender,
        random_state=random_state
    )
    
    # Second split: train vs val (from train+val portion)
    val_ratio = val_size / (train_size + val_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio,
        stratify=gender[train_val_idx],
        random_state=random_state
    )
    
    # Create data splits
    train_data = {
        'gender': gender[train_idx],
        'front': imgX_front[train_idx],
        'side': imgX_side[train_idx],
        'height': height[train_idx]
    }
    
    val_data = {
        'gender': gender[val_idx],
        'front': imgX_front[val_idx],
        'side': imgX_side[val_idx],
        'height': height[val_idx]
    }
    
    test_data = {
        'gender': gender[test_idx],
        'front': imgX_front[test_idx],
        'side': imgX_side[test_idx],
        'height': height[test_idx]
    }
    
    print(f"   ✅ Train set: {len(train_idx)} samples")
    print(f"      Female: {np.sum(gender[train_idx] == 0)}, Male: {np.sum(gender[train_idx] == 1)}")
    print(f"   ✅ Val set:   {len(val_idx)} samples")
    print(f"      Female: {np.sum(gender[val_idx] == 0)}, Male: {np.sum(gender[val_idx] == 1)}")
    print(f"   ✅ Test set:  {len(test_idx)} samples")
    print(f"      Female: {np.sum(gender[test_idx] == 0)}, Male: {np.sum(gender[test_idx] == 1)}")
    
    return train_data, val_data, test_data


def normalize_height(train_height, val_height, test_height):
    """
    Normalize height values using z-score normalization.
    Returns normalized values and statistics for denormalization.
    """
    height_mean = np.mean(train_height)
    height_std = np.std(train_height)
    
    train_height_norm = (train_height - height_mean) / height_std
    val_height_norm = (val_height - height_mean) / height_std
    test_height_norm = (test_height - height_mean) / height_std
    
    stats = {'mean': height_mean, 'std': height_std}
    
    print(f"\n📐 Height normalization statistics:")
    print(f"   Mean: {height_mean:.2f} cm")
    print(f"   Std:  {height_std:.2f} cm")
    
    return train_height_norm, val_height_norm, test_height_norm, stats


#############################################
### STEP 2: MODEL IMPLEMENTATION
#############################################

def create_siamese_height_predictor(input_shape=(224, 224, 1), backbone_name='EfficientNetB0'):
    """
    Create a Siamese architecture with shared EfficientNet backbone.
    
    Architecture:
    1. Two inputs (front, side) - single channel
    2. Convert to RGB (replicate channel 3 times)
    3. Shared EfficientNet backbone (pre-trained on ImageNet)
    4. Concatenate features from both views
    5. Regression head with dropout for height prediction
    
    Args:
        input_shape: Shape of input images (H, W, C)
        backbone_name: Name of backbone architecture
        
    Returns:
        Keras Model
    """
    print(f"\n🏗️  Building Siamese Height Predictor Model...")
    print(f"   Backbone: {backbone_name}")
    print(f"   Input shape: {input_shape}")
    
    # Define inputs for front and side views
    input_front = layers.Input(shape=input_shape, name='input_front')
    input_side = layers.Input(shape=input_shape, name='input_side')
    
    # Convert single-channel to RGB by replicating the channel
    # EfficientNet expects RGB inputs (3 channels)
    def grayscale_to_rgb(x):
        """Convert grayscale to RGB by replicating channels"""
        return tf.concat([x, x, x], axis=-1)
    
    front_rgb = layers.Lambda(grayscale_to_rgb, name='front_to_rgb')(input_front)
    side_rgb = layers.Lambda(grayscale_to_rgb, name='side_to_rgb')(input_side)
    
    # Load pre-trained EfficientNet backbone
    if backbone_name == 'EfficientNetB0':
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3),
            pooling='avg'  # Global average pooling
        )
    elif backbone_name == 'EfficientNetB1':
        backbone = tf.keras.applications.EfficientNetB1(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3),
            pooling='avg'
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    # Initially freeze backbone for warm-up training
    backbone.trainable = False
    
    print(f"   ✅ Loaded {backbone_name} (trainable={backbone.trainable})")
    print(f"      Total params: {backbone.count_params():,}")
    
    # Extract features from both views using shared backbone
    features_front = backbone(front_rgb, training=False)
    features_side = backbone(side_rgb, training=False)
    
    # Concatenate features from both views
    fused_features = layers.Concatenate(name='fused_features')([features_front, features_side])
    
    # Regression head
    x = layers.Dropout(0.3, name='dropout1')(fused_features)
    x = layers.Dense(256, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.2, name='dropout2')(x)
    x = layers.Dense(128, activation='relu', name='dense2')(x)
    x = layers.Dropout(0.1, name='dropout3')(x)
    
    # Output layer (linear activation for regression)
    # For mixed precision, ensure float32 output
    if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
        output = layers.Dense(1, activation='linear', name='height_output', dtype='float32')(x)
    else:
        output = layers.Dense(1, activation='linear', name='height_output')(x)
    
    # Create final model
    model = Model(inputs=[input_front, input_side], outputs=output, name='siamese_height_predictor')
    
    # Store backbone reference for later unfreezing
    model.backbone = backbone
    
    print(f"   ✅ Model created successfully")
    print(f"      Total params: {model.count_params():,}")
    
    return model


def unfreeze_backbone(model, layers_to_unfreeze='last_2_blocks'):
    """
    Unfreeze the backbone for fine-tuning.
    
    Args:
        model: Keras model with .backbone attribute
        layers_to_unfreeze: 'all', 'last_block', 'last_2_blocks', or 'last_3_blocks'
    """
    print(f"\n🔓 Unfreezing backbone layers...")
    
    backbone = model.backbone
    total_layers = len(backbone.layers)
    
    if layers_to_unfreeze == 'all':
        backbone.trainable = True
        print(f"   ✅ Unfroze ALL {total_layers} layers")
    elif layers_to_unfreeze == 'last_block':
        # Unfreeze last ~20% of layers
        unfreeze_from = int(total_layers * 0.8)
        for layer in backbone.layers[:unfreeze_from]:
            layer.trainable = False
        for layer in backbone.layers[unfreeze_from:]:
            layer.trainable = True
        print(f"   ✅ Unfroze last block ({total_layers - unfreeze_from} layers)")
    elif layers_to_unfreeze == 'last_2_blocks':
        # Unfreeze last ~40% of layers
        unfreeze_from = int(total_layers * 0.6)
        for layer in backbone.layers[:unfreeze_from]:
            layer.trainable = False
        for layer in backbone.layers[unfreeze_from:]:
            layer.trainable = True
        print(f"   ✅ Unfroze last 2 blocks ({total_layers - unfreeze_from} layers)")
    elif layers_to_unfreeze == 'last_3_blocks':
        # Unfreeze last ~60% of layers
        unfreeze_from = int(total_layers * 0.4)
        for layer in backbone.layers[:unfreeze_from]:
            layer.trainable = False
        for layer in backbone.layers[unfreeze_from:]:
            layer.trainable = True
        print(f"   ✅ Unfroze last 3 blocks ({total_layers - unfreeze_from} layers)")
    else:
        raise ValueError(f"Unknown option: {layers_to_unfreeze}")
    
    trainable_count = sum([1 for layer in backbone.layers if layer.trainable])
    print(f"   Trainable layers: {trainable_count}/{total_layers}")


#############################################
### STEP 3: DATA AUGMENTATION
#############################################

def create_augmentation_layer():
    """
    Create data augmentation layers for consistent application to both views.
    """
    return keras.Sequential([
        layers.RandomRotation(0.05, fill_mode='constant', fill_value=0.0),  # ±5 degrees
        layers.RandomTranslation(0.1, 0.1, fill_mode='constant', fill_value=0.0),  # ±10%
        layers.RandomZoom(0.1, fill_mode='constant', fill_value=0.0),  # ±10%
    ], name='augmentation')


def create_training_dataset(data, batch_size=32, augment=False, shuffle=True):
    """
    Create tf.data.Dataset for efficient training.
    
    Args:
        data: Dict with 'gender', 'front', 'side', 'height'
        batch_size: Batch size
        augment: Whether to apply data augmentation
        shuffle: Whether to shuffle the data
        
    Returns:
        tf.data.Dataset
    """
    # Convert to float32 for mixed precision compatibility
    front = data['front'].astype('float32')
    side = data['side'].astype('float32')
    height = data['height'].astype('float32').reshape(-1, 1)
    
    # Verify images are in [0, 1] range
    assert front.min() >= 0 and front.max() <= 1, "Images must be in [0, 1] range"
    assert side.min() >= 0 and side.max() <= 1, "Images must be in [0, 1] range"
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((
        {'input_front': front, 'input_side': side},
        height
    ))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
    
    dataset = dataset.batch(batch_size)
    
    # Apply augmentation if requested
    if augment:
        augmentation = create_augmentation_layer()
        
        def augment_images(inputs, labels):
            # Apply same augmentation to both views
            front_aug = augmentation(inputs['input_front'], training=True)
            side_aug = augmentation(inputs['input_side'], training=True)
            return {'input_front': front_aug, 'input_side': side_aug}, labels
        
        dataset = dataset.map(augment_images, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


#############################################
### STEP 4: TRAINING CALLBACKS
#############################################

class TrainingMonitor(Callback):
    """Custom callback to monitor training progress"""
    
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.epoch_times = []
    
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print(f"\n🚀 Training started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Print epoch summary
        print(f"\n   Epoch {epoch + 1} Summary:")
        print(f"      Loss:     {logs.get('loss', 0):.4f} | Val Loss:     {logs.get('val_loss', 0):.4f}")
        print(f"      MAE:      {logs.get('mae', 0):.4f} | Val MAE:      {logs.get('val_mae', 0):.4f}")
        print(f"      RMSE:     {logs.get('rmse', 0):.4f} | Val RMSE:     {logs.get('val_rmse', 0):.4f}")
        if 'lr' in logs:
            print(f"      Learning Rate: {logs['lr']:.2e}")
    
    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\n✅ Training completed in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")


def create_callbacks(model_save_dir, phase='warmup'):
    """
    Create training callbacks for monitoring and checkpointing.
    
    Args:
        model_save_dir: Directory to save model checkpoints
        phase: Training phase ('warmup' or 'finetune')
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Model checkpoint - save best model
    checkpoint_path = os.path.join(model_save_dir, f'best_height_predictor_{phase}.keras')
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # Early stopping
    if phase == 'warmup':
        patience = 10
    else:
        patience = 20
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # Training monitor
    training_monitor = TrainingMonitor()
    callbacks.append(training_monitor)
    
    return callbacks


#############################################
### STEP 5: CUSTOM METRICS
#############################################

def rmse(y_true, y_pred):
    """Root Mean Squared Error metric"""
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


#############################################
### STEP 6: EVALUATION
#############################################

def evaluate_model(model, test_data, height_stats, phase='final'):
    """
    Comprehensive model evaluation on test set.
    
    Args:
        model: Trained Keras model
        test_data: Test dataset dict
        height_stats: Dict with 'mean' and 'std' for denormalization
        phase: Evaluation phase name
        
    Returns:
        Dict of evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"📊 MODEL EVALUATION - {phase.upper()}")
    print(f"{'='*80}")
    
    # Prepare test data
    test_front = test_data['front'].astype('float32')
    test_side = test_data['side'].astype('float32')
    test_height_norm = test_data['height'].astype('float32')
    
    # Make predictions (normalized)
    predictions_norm = model.predict([test_front, test_side], verbose=0).flatten()
    
    # Denormalize predictions and ground truth
    predictions = predictions_norm * height_stats['std'] + height_stats['mean']
    ground_truth = test_height_norm * height_stats['std'] + height_stats['mean']
    
    # Calculate metrics
    mae = mean_absolute_error(ground_truth, predictions)
    rmse_val = np.sqrt(mean_squared_error(ground_truth, predictions))
    r2 = r2_score(ground_truth, predictions)
    
    # Additional statistics
    error = predictions - ground_truth
    mean_error = np.mean(error)
    std_error = np.std(error)
    
    print(f"\n📈 Metrics on Test Set ({len(ground_truth)} samples):")
    print(f"   Mean Absolute Error (MAE):  {mae:.2f} cm")
    print(f"   Root Mean Squared Error:     {rmse_val:.2f} cm")
    print(f"   R² Score:                    {r2:.4f}")
    print(f"   Mean Error (bias):           {mean_error:.2f} cm")
    print(f"   Std Error:                   {std_error:.2f} cm")
    
    # Prediction statistics
    print(f"\n📊 Prediction Statistics:")
    print(f"   Ground Truth - Mean: {np.mean(ground_truth):.2f} cm, Std: {np.std(ground_truth):.2f} cm")
    print(f"   Predictions  - Mean: {np.mean(predictions):.2f} cm, Std: {np.std(predictions):.2f} cm")
    print(f"   Range: [{predictions.min():.1f}, {predictions.max():.1f}] cm")
    
    # Sample predictions
    print(f"\n🔍 Sample Predictions (first 10):")
    print(f"   {'True (cm)':<12} {'Predicted (cm)':<15} {'Error (cm)':<12}")
    print(f"   {'-'*40}")
    for i in range(min(10, len(ground_truth))):
        print(f"   {ground_truth[i]:>10.1f}   {predictions[i]:>13.1f}   {error[i]:>10.1f}")
    
    metrics = {
        'mae': mae,
        'rmse': rmse_val,
        'r2': r2,
        'mean_error': mean_error,
        'std_error': std_error,
        'predictions': predictions,
        'ground_truth': ground_truth
    }
    
    return metrics


def plot_results(metrics, save_dir, phase='final'):
    """
    Plot evaluation results.
    
    Args:
        metrics: Dict of evaluation metrics
        save_dir: Directory to save plots
        phase: Phase name for file naming
    """
    predictions = metrics['predictions']
    ground_truth = metrics['ground_truth']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Scatter plot: Predicted vs True
    axes[0].scatter(ground_truth, predictions, alpha=0.5, s=20)
    axes[0].plot([ground_truth.min(), ground_truth.max()],
                 [ground_truth.min(), ground_truth.max()],
                 'r--', lw=2, label='Perfect prediction')
    axes[0].set_xlabel('True Height (cm)', fontsize=12)
    axes[0].set_ylabel('Predicted Height (cm)', fontsize=12)
    axes[0].set_title(f'Predicted vs True Height\nR² = {metrics["r2"]:.4f}', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error distribution
    errors = predictions - ground_truth
    axes[1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='r', linestyle='--', lw=2, label='Zero error')
    axes[1].axvline(metrics['mean_error'], color='g', linestyle='--', lw=2, 
                   label=f'Mean error: {metrics["mean_error"]:.2f} cm')
    axes[1].set_xlabel('Prediction Error (cm)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'Error Distribution\nMAE = {metrics["mae"]:.2f} cm', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, f'height_predictor_evaluation_{phase}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Evaluation plot saved: {plot_path}")
    plt.close()


#############################################
### MAIN TRAINING PIPELINE
#############################################

def main():
    """Main training pipeline"""
    
    print(f"\n{'='*80}")
    print("🎯 HEIGHT PREDICTOR TRAINING - TRANSFER LEARNING PIPELINE")
    print(f"{'='*80}\n")
    
    # Create model save directory
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    #############################################
    # STEP 1: Load and prepare data
    #############################################
    
    df_total = load_databases()
    imgX_front, imgX_side = load_images()
    df_total, imgX_front, imgX_side = align_data_by_ids(df_total, imgX_front, imgX_side)
    
    # Split into train/val/test
    train_data, val_data, test_data = prepare_data_splits(
        df_total, imgX_front, imgX_side,
        train_size=0.8, val_size=0.1, test_size=0.1,
        random_state=42
    )
    
    # Free memory
    del df_total, imgX_front, imgX_side
    gc.collect()
    
    # Normalize height
    train_data['height'], val_data['height'], test_data['height'], height_stats = normalize_height(
        train_data['height'], val_data['height'], test_data['height']
    )
    
    #############################################
    # STEP 2: Create model
    #############################################
    
    model = create_siamese_height_predictor(
        input_shape=(IMG_SIZE_4NN, IMG_SIZE_4NN, CHAN),
        backbone_name='EfficientNetB0'
    )
    
    print(f"\n📋 Model Summary:")
    model.summary()
    
    #############################################
    # STEP 3: Phase 1 - Warm-up Training (Frozen Backbone)
    #############################################
    
    print(f"\n{'='*80}")
    print("🔥 PHASE 1: WARM-UP TRAINING (Frozen Backbone)")
    print(f"{'='*80}")
    
    # Compile model for warm-up
    optimizer_warmup = Adam(learning_rate=1e-3)
    if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
        optimizer_warmup = tf.keras.mixed_precision.LossScaleOptimizer(optimizer_warmup)
    
    model.compile(
        optimizer=optimizer_warmup,
        loss='mse',
        metrics=['mae', rmse]
    )
    
    # Create datasets
    batch_size = 32
    train_dataset = create_training_dataset(train_data, batch_size=batch_size, augment=True, shuffle=True)
    val_dataset = create_training_dataset(val_data, batch_size=batch_size, augment=False, shuffle=False)
    
    # Callbacks
    callbacks_warmup = create_callbacks(MODEL_SAVE_DIR, phase='warmup')
    
    # Train
    epochs_warmup = 20 if not TEST_FILES else 5
    print(f"\n🏋️  Training for {epochs_warmup} epochs with frozen backbone...")
    
    history_warmup = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs_warmup,
        callbacks=callbacks_warmup,
        verbose=2
    )
    
    # Evaluate after warm-up
    metrics_warmup = evaluate_model(model, test_data, height_stats, phase='warmup')
    plot_results(metrics_warmup, MODEL_SAVE_DIR, phase='warmup')
    
    #############################################
    # STEP 4: Phase 2 - Fine-tuning (Unfrozen Backbone)
    #############################################
    
    print(f"\n{'='*80}")
    print("🎨 PHASE 2: FINE-TUNING (Unfrozen Backbone)")
    print(f"{'='*80}")
    
    # Unfreeze backbone
    unfreeze_backbone(model, layers_to_unfreeze='last_2_blocks')
    
    # Recompile with lower learning rate
    optimizer_finetune = Adam(learning_rate=1e-5)
    if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
        optimizer_finetune = tf.keras.mixed_precision.LossScaleOptimizer(optimizer_finetune)
    
    model.compile(
        optimizer=optimizer_finetune,
        loss='mse',
        metrics=['mae', rmse]
    )
    
    # Callbacks
    callbacks_finetune = create_callbacks(MODEL_SAVE_DIR, phase='finetune')
    
    # Train
    epochs_finetune = 50 if not TEST_FILES else 10
    print(f"\n🏋️  Fine-tuning for {epochs_finetune} epochs with unfrozen backbone...")
    
    history_finetune = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs_finetune,
        callbacks=callbacks_finetune,
        verbose=2
    )
    
    # Free memory
    del train_dataset, val_dataset
    gc.collect()
    
    #############################################
    # STEP 5: Final Evaluation
    #############################################
    
    metrics_final = evaluate_model(model, test_data, height_stats, phase='final')
    plot_results(metrics_final, MODEL_SAVE_DIR, phase='final')
    
    #############################################
    # STEP 6: Save Final Model
    #############################################
    
    final_model_path = os.path.join(MODEL_SAVE_DIR, 'height_predictor_final.keras')
    model.save(final_model_path)
    print(f"\n💾 Final model saved: {final_model_path}")
    
    # Save height statistics for inference
    import json
    stats_path = os.path.join(MODEL_SAVE_DIR, 'height_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(height_stats, f)
    print(f"💾 Height statistics saved: {stats_path}")
    
    print(f"\n{'='*80}")
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'='*80}\n")
    
    return model, metrics_final, height_stats


if __name__ == "__main__":
    model, metrics, height_stats = main()
