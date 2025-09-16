# Extractor CNN + MLP Training Documentation

## 1. Purpose
The extractor predicts a set of unknown anthropometric measurements (`UK_MEAS`) from:
- Two binary silhouette images (front + side)
- A small numerical/categorical feature vector (`KN_MEAS`, e.g. gender, height)

Task type: Multivariate regression → outputs are continuous standardized measurement values later de-normalized for downstream stages (imputation / 3D synthesis).

---
## 2. End-to-End Training Flow
```
Load tabular measurements  →  Load silhouettes (front & side)  →  Split train/test
     ↓                                  ↓                           ↓
Normalize targets (global stats)   Prepare image arrays       Scale numerical inputs
     ↓                                  ↓                           ↓
Build MLP branch                Build shared CNN branch (reused for both views)
     ↓                                  ↓
                  Concatenate latent outputs → Fusion dense → Regression head
                                      ↓
         Multi-input generator (augmented images + numeric features)
                                      ↓
                    Train (MSE loss, Adam, EarlyStopping)
                                      ↓
                  Save model, scaler(s), training history plot
```

---
## 3. Data Sources & Assembly
### 3.1 Anthropometric Tables
Function: `load_databases()`
- Integrates three datasets: SPRING, ANSUR I, ANSUR II
- Processes each gender separately; retains only `UK_MEAS + KN_MEAS`
- Adds numeric gender flag
- Concatenates female group, male group, then merges → `df_total`
- Test mode optionally truncates for rapid prototyping

### 3.2 Silhouette Images
Function: `load_images()`
- Loads precomputed NumPy silhouette tensors per dataset, gender, and view
- Stacks (SPRING + ANSUR I + ANSUR II) for each gender/view
- Concatenates genders → final arrays:
  - `imgX_front.shape == (N_total, H, W, CHAN)`
  - `imgX_side.shape  == (N_total, H, W, CHAN)`
- Frees intermediate arrays (explicit `del` + `gc.collect()`) to conserve RAM

### 3.3 Train/Test Partition
```
(trainData, testData,
 trainImgXf, testImgXf,
 trainImgXs, testImgXs) = train_test_split(
    df_total, imgX_front, imgX_side,
    train_size=0.8, shuffle=True, random_state=123)
```
A single synchronized split preserves alignment of tabular & both image modalities.

---
## 4. Feature & Label Processing
### 4.1 Target Normalization
```
trainMeasY = (trainData[UK_MEAS] - df_total[UK_MEAS].mean()) / df_total[UK_MEAS].std()
testMeasY  = (testData[UK_MEAS]  - df_total[UK_MEAS].mean()) / df_total[UK_MEAS].std()
```
Note: Uses global (train+test) statistics → introduces mild data leakage (see Improvements).

### 4.2 Numerical Input Scaling
`process_db_values()` returns scaled `(trainMeasX, testMeasX)` plus scaler object (Standard or MinMax). If gender later excluded it is removed column-wise. Sets global `INP_SHAPE` for MLP input dimension.

### 4.3 Memory Cleanup
Original large image arrays are deleted after splitting; only split subsets remain.

---
## 5. Image Data Augmentation
Configured in `main()` via `ImageDataGenerator`:
```
featurewise_center = True
featurewise_std_normalization = True
width_shift_range = 0.10
height_shift_range = 0.25
shear_range = 3
zoom_range = [0.8, 1.2]
horizontal_flip = False
```
`datagen.fit(trainImgXf, augment=True)` computes mean/std for feature-wise normalization (front images only).

Rationale:
- Centering & std normalization unify intensity distribution across datasets.
- Shifts & zoom simulate framing variability.
- Shear approximates slight camera angle deviations.
- No flipping preserves anatomical left/right fidelity.

---
## 6. Model Architecture
### 6.1 MLP Branch (`createMLP_model`)
Input: `(INP_SHAPE,)`
Layers:
1. Dense(16, relu)
2. Inner loop: `in_MLPlayers` × Dense(64, relu)
3. Dense(64, relu)
4. Output proxy: Dense(len(UK_MEAS), linear)

Produces a target-dimensional latent embedding from numerical & categorical context.

### 6.2 CNN Branch (`createCNN_model`)
Input: `(IMG_SIZE_4NN, IMG_SIZE_4NN, CHAN)`
Layers:
1. Conv2D(96, 5×5, relu) → MaxPool(3×3)
2. Loop: `in_CNNlayers` × Conv2D(128, 3×3, relu)
3. Flatten
4. Dense(200, relu) → Dropout(0.5)
5. Output proxy: Dense(len(UK_MEAS), linear)

This CNN instance is weight-shared: applied to front and side silhouettes separately to enforce view-invariant feature learning.

### 6.3 Fusion (`createCombined_model`)
Inputs: numerical proxy + front proxy + side proxy
```
concatenate → Dense(2×len(UK_MEAS), relu) → Dense(len(UK_MEAS), linear)
```
Final output: normalized predicted measurements.

Design Notes:
- Late fusion allows each modality to specialize before integration.
- Target-sized proxy heads encourage supervised shaping of intermediate embeddings.
- Weight tying reduces parameters & overfitting risk given moderate dataset size.

---
## 7. Multi-Input Data Generator
`generator2imgsNumData(datagen, MeasX, ImgXf, ImgXs, MeasY, batch_size)`
- Creates three internal flows (numeric, front, side) with deterministic seeds.
- Yields: `([numeric_batch, front_batch, side_batch], y_batch)`
- Front & side batches are independently augmented; numerical data is not.

Caveat: Indexing uses tuple positions like `Xni[1]`, `Xfi[0]`, etc.—sensitive to Keras `flow` return structure (improvement candidate).

---
## 8. Training Configuration
- Optimizer: `Adam(learning_rate=1e-4)`
- Loss: `mse`
- Metric: `mean_absolute_error`
- Batch size: 32
- Max epochs: 500 (or 20 in test mode)
- EarlyStopping: `monitor=val_loss`, `patience=10`, `min_delta=1e-4`, `restore_best_weights=True`
- Steps:
  - `steps_per_epoch = floor(train_samples / batch_size)`
  - `validation_steps = floor(test_samples / batch_size)`

The effective number of epochs trained (pre-early-stop) is recorded via history length and embedded in model filename.

---
## 9. Artifact Persistence
- Model saved as: `extractor_{scalerType}_img{IMG_SIZE_4NN}_inp{len(KN_MEAS)}_out{len(UK_MEAS)}_ep{final_epochs}[_{TEST_FILES_NUM}test].keras`
- Scaler saved separately (Standard vs MinMax) → e.g. `scalerStd_extractor.pkl`
- Training curve plotted with `histplot(history, metric="mean_absolute_error")`

Missing (potential improvements): explicit persistence of label normalization mean/std in a JSON or pkl.

---
## 10. Inference Path (Conceptual)
1. Load saved model + input scaler + stored label stats.
2. Preprocess numerical inputs with scaler.
3. Preprocess silhouettes: resize → convert to expected shape → apply feature centering/std using stored `datagen` stats (if serialized) or replicate pipeline.
4. Predict normalized outputs.
5. De-normalize: `pred * std + mean` (must match training statistics order of `UK_MEAS`).

---
## 11. Design Strengths
- Hybrid modality exploitation (context + morphology).
- Weight sharing reduces overfitting and data requirement.
- Augmentation tailored to real camera variability without anatomically implausible transforms.
- Early stopping + low LR for stable convergence.
- Encapsulation of each architectural piece (MLP, CNN, fusion) improves maintainability.

---
## 12. Limitations / Risks
| Area | Issue | Impact |
|------|-------|--------|
| Label normalization | Uses global stats (train+test) | Minor optimistic bias |
| Augmentation stats | Fitted only on front images | Slight distribution mismatch for side |
| Generator | Relies on tuple index order | Fragile if Keras version changes |
| Measurement weighting | All targets equal | High-variance measurements may dominate MSE |
| Reproducibility | No consolidated seeding across libs | Run-to-run variation |
| Metrics | Only aggregate MAE | No per-measurement diagnostics |
| Checkpointing | Only EarlyStopping restore | No separate best-loss archive |
| Scaling persistence | Label mean/std not saved | Harder reproducible inference |

---
## 13. Recommended Improvements
1. Train-only normalization for targets; persist `{'mean': list, 'std': list}` JSON.
2. Fit augmentation stats on concatenated front+side: `datagen.fit(np.concatenate([trainImgXf, trainImgXs], axis=0))`.
3. Replace generator with `tf.keras.utils.Sequence` class for clarity & deterministic length.
4. Introduce per-output MAE callback:
   - Custom Keras callback computing column-wise MAE each epoch → CSV log.
5. Add learning rate scheduler: `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)`.
6. Measurement weighting: `loss = tf.keras.losses.MSE` wrapped with weights vector (e.g. inverse variance).
7. Add reproducibility block:
   ```python
   import os, random, numpy as np, tensorflow as tf
   os.environ['PYTHONHASHSEED']='123'
   random.seed(123); np.random.seed(123); tf.random.set_seed(123)
   ```
8. Provide inference helper utility (`predict_measurements(front, side, num_features)`).
9. Version the model artifact names with semantic versioning or git hash for traceability.
10. Log training with lightweight experiment tracker (optional): CSV + TensorBoard at minimum.

---
## 14. Quick Reference Cheat Sheet
| Component | Function | Key Args |
|-----------|----------|----------|
| Data loader | `load_databases` | Consolidates 3 datasets, filters cols |
| Image loader | `load_images` | Loads & stacks silhouettes |
| Scaling | `process_db_values` | Applies numerical scaler |
| MLP | `createMLP_model` | `in_MLPlayers` inner Dense(64)s |
| CNN | `createCNN_model` | Shared for front & side |
| Fusion | `createCombined_model` | Concatenate + Dense(2×targets) |
| Generator | `generator2imgsNumData` | Multi-input batch yield |
| Training | `model.fit` | MSE + Adam(1e-4) + EarlyStopping |
| Persistence | `.save()` | Filename encodes config |

---
## 15. Minimal Inference Pseudocode
```python
# Load artifacts
model = keras.models.load_model(model_path)
num_scaler = joblib.load(num_scaler_path)
label_stats = json.load(open('label_norm_stats.json'))  # {'mean': [...], 'std': [...]} if added

# Prepare inputs
def prep_sil(img):
    # resize to IMG_SIZE_4NN, expand dims, apply (x - feat_mean)/feat_std
    ...

front_proc = prep_sil(front_img)
side_proc  = prep_sil(side_img)
num_input  = num_scaler.transform([raw_num_features])

# Predict
pred_norm = model.predict([num_input, front_proc, side_proc])

# De-normalize
pred = pred_norm * label_stats['std'] + label_stats['mean']
```

---
## 16. Summary
The extractor employs a late-fusion hybrid CNN+MLP architecture with shared convolutional weights across dual silhouette views and supervised proxy heads to stabilize multi-modal regression. A tailored augmentation regime and conservative optimization settings promote generalization on heterogeneous anthropometric datasets. Several incremental enhancements (train-only normalization, structured generators, richer metrics) can further harden the pipeline for production and research reproducibility.
