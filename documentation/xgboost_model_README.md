# XGBoost Body Measurement Prediction Model

## Overview

This regression model uses XGBoost (Extreme Gradient Boosting) to predict hard-to-measure body measurements from easily obtainable proxy measurements. The model is trained on the CAESAR anthropometric dataset.

## Model Performance

The trained XGBoost model achieves excellent performance:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Average MAE** | 4.12 | Predictions are off by ~4.12 units on average |
| **Average RMSE** | 5.42 | Root mean squared error across all targets |
| **Average R²** | 0.832 | Model explains 83.2% of the variance |

### Performance by Target Measurement

| Target Measurement | MAE | RMSE | R² Score |
|-------------------|-----|------|----------|
| Triceps Skinfold | 4.26 | 5.66 | 0.722 |
| Subscapular Skinfold | 4.58 | 6.06 | 0.723 |
| Supraspinale Skinfold | 4.59 | 5.94 | 0.742 |
| Calf Skinfold | 2.74 | 3.74 | 0.890 |
| Humerus Breadth | 1.13 | 1.43 | 0.857 |
| Femur Breadth | 1.66 | 2.08 | 0.885 |
| Arm Circumference (Flexed) | 6.18 | 7.89 | 0.919 |
| Calf Circumference | 7.83 | 10.57 | 0.917 |

## How XGBoost Works

### Conceptual Analogy: A Team of Specialists Learning from Mistakes

Imagine a team of experts working in sequence:

1. **First Expert (Rookie)**: Makes an initial guess at the measurements. Their guess will have some errors.

2. **Second Expert**: Looks at the first expert's mistakes (the "residuals") and predicts those mistakes.

3. **Combination**: You combine the first expert's guess with the second expert's correction to get a better prediction.

4. **Third Expert**: Looks at the new, smaller mistakes and predicts them.

This process repeats, with each new specialist focusing on correcting the remaining errors of the team. XGBoost does this with decision trees, building a powerful, sequential chain where each new tree improves upon the last.

### Technical Process

1. **Sequential Tree Building**: Unlike Random Forest (which builds trees in parallel), XGBoost builds them one after another.

2. **Learning from Residuals**: The first tree is fit to the data. The model then calculates the errors (residuals) between its predictions and the actual values.

3. **Targeting Errors**: The second tree is trained to predict the errors made by the first tree.

4. **Additive Combination**: The prediction from the second tree (error correction) is scaled by a learning rate and added to the first tree's prediction.

5. **Iteration**: This process repeats for hundreds of trees, with each new tree focusing on the ever-shrinking errors, gradually refining the model into a highly precise predictor.

## Model Architecture

### Input Features (Proxy Measurements)

The model uses 12 easily obtainable measurements as inputs:

1. **Stature** - Height (mm)
2. **Weight** - Body weight (kg)
3. **Chest Circumference** - Chest measurement (mm)
4. **Hip Circumference** - Hip measurement (mm)
5. **Waist Circumference** - Waist measurement (mm)
6. **Thigh Circumference** - Thigh measurement (mm)
7. **Ankle Circumference** - Ankle measurement (mm)
8. **Shoulder Breadth** - Shoulder width (mm)
9. **Knee Height** - Knee height (mm)
10. **BMI** - Body Mass Index
11. **Age** - Age in years
12. **Gender_Male** - 1 for male, 0 for female

### Output Targets (Predicted Measurements)

The model predicts 8 harder-to-measure body parameters:

1. **Triceps Skinfold** - Skinfold thickness at triceps (mm)
2. **Subscapular Skinfold** - Skinfold at subscapular area (mm)
3. **Supraspinale Skinfold** - Skinfold at supraspinale (mm)
4. **Calf Skinfold** - Skinfold at calf (mm)
5. **Humerus Breadth** - Humerus bone breadth (mm)
6. **Femur Breadth** - Femur bone breadth (mm)
7. **Arm Circumference (Flexed)** - Flexed arm circumference (mm)
8. **Calf Circumference** - Calf circumference (mm)

### XGBoost Hyperparameters

```python
n_estimators=1000          # Number of boosting rounds (trees)
learning_rate=0.05         # Step size shrinkage (eta)
max_depth=6                # Maximum depth of trees
subsample=0.8              # Fraction of samples per tree
colsample_bytree=0.8       # Fraction of features per tree
min_child_weight=1         # Minimum sum of instance weight
gamma=0                    # Minimum loss reduction for split
reg_alpha=0.1              # L1 regularization
reg_lambda=1.0             # L2 regularization
```

## Usage

### Training a New Model

```python
from src.predictormodel import BodyMeasurementPredictor

# Initialize predictor
predictor = BodyMeasurementPredictor()

# Define measurements
predictor.define_measurements()

# Load data
df = predictor.load_data('path/to/CAESAR_full_imputation_dataset.csv')

# Prepare and split data
X, y = predictor.prepare_data(df)
X_train, X_test, y_train, y_test = predictor.split_data(X, y)

# Build and train
predictor.build_model()
predictor.train(X_train, y_train)

# Evaluate
results = predictor.evaluate(X_test, y_test)

# Save model
predictor.save_model()
```

### Making Predictions

```python
from src.predictormodel import BodyMeasurementPredictor
import pandas as pd

# Load trained model
predictor = BodyMeasurementPredictor()
predictor.load_model('path/to/saved_model.pkl')

# Prepare input data
person_data = {
    'Stature': 1750,
    'Weight': 75.0,
    'Chest_Circumference': 950,
    'Hip_Circumference': 980,
    'Waist_Circumference': 850,
    'Thigh_Circumference': 580,
    'Ankle_Circumference': 230,
    'Shoulder_Breadth': 420,
    'Knee_Height': 500,
    'BMI': 24.5,
    'Age': 30,
    'Gender_Male': 1
}

X = pd.DataFrame([person_data])

# Make prediction
predictions = predictor.predict(X)
print(predictions)
```

### Running the Complete Training Script

```bash
python src/predictormodel.py
```

### Running Example Usage Scripts

```bash
python src/use_predictor.py
```

## Strengths of XGBoost for This Project

1. **State-of-the-Art Performance**: XGBoost is renowned for achieving the highest accuracy on structured (tabular) data, making it a top choice for anthropometric prediction.

2. **High Speed and Efficiency**: Highly optimized library designed for performance.

3. **Built-in Regularization**: Includes techniques to prevent overfitting, leading to better generalization on new subjects.

4. **Handles Complex Patterns**: Can capture non-linear relationships between input and output measurements.

5. **Robust to Outliers**: The boosting process makes it resilient to outliers in the data.

## Dataset Information

- **Source**: CAESAR (Civilian American and European Surface Anthropometry Resource)
- **Total Samples**: 2,388 subjects
- **Training Split**: 1,910 samples (80%)
- **Testing Split**: 478 samples (20%)
- **Features**: 23 total measurements in the dataset
- **Missing Values**: Handled by median imputation

## Files

- `src/predictormodel.py` - Main model implementation and training script
- `src/use_predictor.py` - Example usage and prediction scripts
- `data/datasets/CAESAR_full_imputation_dataset.csv` - Training dataset
- `data/model_files/` - Directory for saved models

## Requirements

```
pandas>=2.0.3
numpy>=1.24.3
scikit-learn>=1.3.0
xgboost>=3.0.0
```

## Future Improvements

1. **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV to optimize parameters
2. **Feature Engineering**: Create interaction features or polynomial features
3. **Cross-Validation**: Implement k-fold cross-validation for more robust evaluation
4. **Model Ensembling**: Combine XGBoost with other models (e.g., Random Forest, Neural Networks)
5. **SHAP Values**: Add interpretability using SHAP (SHapley Additive exPlanations)

## License

Please refer to the main project LICENSE file.

## References

- XGBoost: A Scalable Tree Boosting System (Chen & Guestrin, 2016)
- CAESAR Database: http://store.sae.org/caesar/
