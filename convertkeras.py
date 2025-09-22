import os, sys, time, argparse, csv
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow import keras

# make src importable
sys.path.append("src")
from extractor.extractor_model_training import (
    load_databases,
    load_images,
    align_data_by_ids,
    process_db_values,
)
from utils import MODEL_FILES_DIR, DS_DIR, UK_MEAS

# Serializable Cast shim for mixed-precision models
class CastLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)
    def get_config(self):
        return {}

tf.keras.utils.get_custom_objects()["Cast"] = CastLayer

def load_model_safe(path):
    # choose proper custom_objects when loading mixed-precision models
    return keras.models.load_model(path, compile=False, custom_objects={"Cast": CastLayer})

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.join("data","model_files","extractor_stdScal_img224_inp2_out12_ep195.keras"))
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--out", default="benchmark_results.csv")
    args = p.parse_args()

    print("Loading model:", args.model)
    model = load_model_safe(args.model)

    print("Loading databases and images (same pipeline as training)...")
    df_total = load_databases()
    imgX_front, imgX_side = load_images()
    df_total, imgX_front, imgX_side = align_data_by_ids(df_total, imgX_front, imgX_side)

    # Recreate the training/test split exactly as in training script
    from sklearn.model_selection import train_test_split
    trainData, testData, trainImgXf, testImgXf, trainImgXs, testImgXs = train_test_split(
        df_total, imgX_front, imgX_side, train_size=0.8, shuffle=True, random_state=123
    )

    # Build numeric inputs using same scaler pipeline as training
    (trainMeasX, testMeasX), scaler = process_db_values(df_total, trainData, testData)

    # Remove gender column if training removed it
    from utils import KN_MEAS
    if "gender" not in KN_MEAS:
        testMeasX = np.delete(testMeasX, 0, 1)

    # Convert to float32 as training did
    testMeasX = testMeasX.astype("float32")
    testImgXf = testImgXf.astype("float32")
    testImgXs = testImgXs.astype("float32")

    # Prepare true labels (same normalization used in training)
    testMeasY = (testData[UK_MEAS] - df_total[UK_MEAS].mean()) / df_total[UK_MEAS].std()
    y_true = testMeasY.values.astype("float32")

    # Warm-up and timing
    print("Warming up model...")
    _ = model.predict([testMeasX[:8], testImgXf[:8], testImgXs[:8]], batch_size=args.batch, verbose=0)

    print("Running inference on test set...")
    t0 = time.perf_counter()
    preds = model.predict([testMeasX, testImgXf, testImgXs], batch_size=args.batch, verbose=0)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    throughput = testMeasX.shape[0] / elapsed

    overall_mae = mean_absolute_error(y_true, preds)
    per_output_mae = np.mean(np.abs(preds - y_true), axis=0)

    print(f"Samples: {testMeasX.shape[0]}  Time: {elapsed:.3f}s  Throughput: {throughput:.1f} samples/s")
    print(f"Overall MAE: {overall_mae:.4f}")
    for i, m in enumerate(per_output_mae):
        print(f"Output[{i}] MAE: {m:.4f}")

    # Save results CSV
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["samples", testMeasX.shape[0]])
        w.writerow(["time_s", elapsed])
        w.writerow(["throughput_samples_per_s", throughput])
        w.writerow(["overall_mae", overall_mae])
        w.writerow(["per_output_mae"] + per_output_mae.tolist())

    print("Results saved to", args.out)

if __name__ == "__main__":
    main()