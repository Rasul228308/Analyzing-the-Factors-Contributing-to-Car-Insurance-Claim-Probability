# Car Insurance Risk Analysis - Function Documentation

## 📊 Project Context
**Dataset**: Annual car insurance data with 19 features  
**Target Variable**: `OUTCOME` (1 = customer filed claim, 0 = no claim)  
**Objective**: Predict customer claim probability using machine learning

---

## 📋 Table of Contents
1. [Data Loading & Exploration](#1-data-loading--exploration)
2. [Missing Data Analysis](#2-missing-data-analysis)
3. [Feature Engineering](#3-feature-engineering)
4. [Data Preprocessing](#4-data-preprocessing)
5. [Statistical Analysis](#5-statistical-analysis)
6. [Model Training](#6-model-training)
7. [Model Evaluation](#7-model-evaluation)
8. [Hyperparameter Optimization](#8-hyperparameter-optimization)
9. [Model Diagnostics](#9-model-diagnostics)
10. [Calibration](#10-calibration)
11. [Post-Hoc Analysis](#11-post-hoc-analysis)
12. [Business Analytics](#12-business-analytics)

---

## 1. Data Loading & Exploration

### `load_data(filepath, make_copy=True)`
**Purpose**: Load CSV data and optionally create a backup copy

**Inputs**:
- `filepath` (str): Path to CSV file
- `make_copy` (bool, default=True): Whether to create a copy of the dataframe

**Outputs**:
- `df` (DataFrame): Original dataframe
- `df_copy` (DataFrame or None): Copy of dataframe if make_copy=True

**Example**:
```python
df, df_backup = load_data('insurance_data.csv')
```

---

### `display_basic_info(df)`
**Purpose**: Display comprehensive dataset overview

**Inputs**:
- `df` (DataFrame): Input dataframe

**Outputs**:
- Prints: Dataset info, first 5 rows, descriptive statistics
- Returns: Tuple of (df.info() result, df.describe() result)

**Example**:
```python
info, stats = display_basic_info(df)
```

---

## 2. Missing Data Analysis

### `analyze_missing_data(df, missing_cols=['CREDIT_SCORE', 'ANNUAL_MILEAGE'])`
**Purpose**: Report missing data percentages for specified columns

**Inputs**:
- `df` (DataFrame): Input dataframe
- `missing_cols` (list, default=['CREDIT_SCORE', 'ANNUAL_MILEAGE']): Columns to analyze

**Outputs**:
- Prints: Missing data report with percentages
- Returns: Dictionary {column: missing_count}

**Example**:
```python
missing_report = analyze_missing_data(df)
# Output: {'CREDIT_SCORE': 274, 'ANNUAL_MILEAGE': 446}
```

---

### `analyze_missingness_mechanism(df, target_vars, predictor_vars)`
**Purpose**: Determine if data is Missing At Random (MAR) or Missing Completely At Random (MCAR)

**Inputs**:
- `df` (DataFrame): Input dataframe
- `target_vars` (list, default=['CREDIT_SCORE', 'ANNUAL_MILEAGE']): Variables with missing values
- `predictor_vars` (list, default=['VEHICLE_YEAR', 'DRIVING_EXPERIENCE', 'VEHICLE_TYPE']): Variables to test association

**Outputs**:
- Returns: DataFrame with Chi-square test results (Chi2, p-value, Mechanism)

**Interpretation**:
- p-value < 0.05 → MAR (missing data depends on other variables)
- p-value ≥ 0.05 → MCAR (missing data is random)

**Example**:
```python
mechanism_df = analyze_missingness_mechanism(df)
```

---

### `plot_chi2_heatmap(summary_df, figsize=(10, 6))`
**Purpose**: Visualize missingness mechanism test results

**Inputs**:
- `summary_df` (DataFrame): Output from `analyze_missingness_mechanism()`
- `figsize` (tuple, default=(10, 6)): Figure size

**Outputs**:
- Displays: Heatmap of p-values

**Example**:
```python
plot_chi2_heatmap(mechanism_df)
```

---

## 3. Feature Engineering

### `encode_ordinal_features(df, ordinal_mapping=None)`
**Purpose**: Encode ordinal categorical variables with meaningful numeric order

**Inputs**:
- `df` (DataFrame): Input dataframe
- `ordinal_mapping` (dict, optional): Custom mapping (default uses predefined mappings)

**Default Mappings**:
```python
{
    'DRIVING_EXPERIENCE': {'0-9y': 0, '10-19y': 1, '20-29y': 2, '30y+': 3},
    'EDUCATION': {'none': 0, 'high school': 1, 'university': 2},
    'INCOME': {'poverty': 0, 'working class': 1, 'middle class': 2, 'upper class': 3},
    'VEHICLE_YEAR': {'before 2015': 0, 'after 2015': 1},
    'AGE': {'16-25': 0, '26-39': 1, '40-64': 2, '65+': 3}
}
```

**Outputs**:
- Returns: Modified dataframe with encoded columns (int8 dtype)

**Example**:
```python
df = encode_ordinal_features(df)
```

---

### `encode_binary_features(df, binary_vars)`
**Purpose**: Encode binary/nominal categorical variables using LabelEncoder

**Inputs**:
- `df` (DataFrame): Input dataframe
- `binary_vars` (list, default=['GENDER', 'VEHICLE_TYPE', 'VEHICLE_OWNERSHIP', 'MARRIED', 'CHILDREN', 'RACE']): Variables to encode

**Outputs**:
- Returns: Modified dataframe with encoded columns (int8 dtype)

**Example**:
```python
df = encode_binary_features(df)
```

---

### `encode_categorical_features(df, cat_vars=['RACE'])`
**Purpose**: Alternative encoding for specific categorical variables

**Inputs**:
- `df` (DataFrame): Input dataframe
- `cat_vars` (list, default=['RACE']): Variables to encode

**Outputs**:
- Returns: Modified dataframe with encoded columns (int8 dtype)

---

### `optimize_numeric_dtypes(df)`
**Purpose**: Reduce memory usage by optimizing data types

**Inputs**:
- `df` (DataFrame): Input dataframe

**Outputs**:
- Returns: Modified dataframe with optimized dtypes:
  - ID, POSTAL_CODE → int32
  - SPEEDING_VIOLATIONS, DUIS, PAST_ACCIDENTS, OUTCOME → int8
  - CREDIT_SCORE, ANNUAL_MILEAGE → float32

**Memory Savings**: ~50-70% reduction in memory usage

**Example**:
```python
df = optimize_numeric_dtypes(df)
```

---

## 4. Data Preprocessing

### `drop_missing_values(df, subset_cols, check_outcome=True)`
**Purpose**: Drop rows with missing values and verify target preservation

**Inputs**:
- `df` (DataFrame): Input dataframe
- `subset_cols` (list, default=['CREDIT_SCORE', 'ANNUAL_MILEAGE']): Columns to check for nulls
- `check_outcome` (bool, default=True): Verify OUTCOME distribution is preserved

**Outputs**:
- Prints: Shape before/after, records dropped, target preservation check
- Returns: Cleaned dataframe

**Example**:
```python
df_clean = drop_missing_values(df)
# Output:
# Dataset shape before: (10000, 19)
# Records dropped: 720 (7.2%)
# Claim rate BEFORE: 0.2615
# Claim rate AFTER: 0.2608
# ✅ Preserved
```

---

## 5. Statistical Analysis

### `test_normality(df, numeric_cols, sample_size=5000)`
**Purpose**: Test if numeric variables follow normal distribution

**Inputs**:
- `df` (DataFrame): Input dataframe
- `numeric_cols` (list, default=['CREDIT_SCORE', 'ANNUAL_MILEAGE']): Columns to test
- `sample_size` (int, default=5000): Maximum sample size for Shapiro-Wilk test

**Outputs**:
- Prints: Test results for each variable
- Returns: DataFrame with Shapiro-Wilk and Kolmogorov-Smirnov test statistics

**Interpretation**:
- p-value > 0.05 → Normal distribution
- p-value ≤ 0.05 → Non-normal distribution

**Example**:
```python
normality_df = test_normality(df)
```

---

### `plot_diagnostic_plots(df, numeric_cols)`
**Purpose**: Visualize distribution, normality, and outliers

**Inputs**:
- `df` (DataFrame): Input dataframe
- `numeric_cols` (list): Columns to plot

**Outputs**:
- Displays: 3 plots per variable:
  1. Histogram + KDE
  2. Q-Q plot vs Normal distribution
  3. Boxplot (outlier detection)

**Example**:
```python
plot_diagnostic_plots(df, ['CREDIT_SCORE', 'ANNUAL_MILEAGE'])
```

---

### `plot_correlation_heatmap(df, target_col='OUTCOME', figsize=(6, 10))`
**Purpose**: Show feature correlations with target variable

**Inputs**:
- `df` (DataFrame): Input dataframe
- `target_col` (str, default='OUTCOME'): Target variable name
- `figsize` (tuple, default=(6, 10)): Figure size

**Outputs**:
- Displays: Heatmap of correlations sorted by strength
- Returns: Correlation series sorted by target correlation

**Example**:
```python
correlations = plot_correlation_heatmap(df)
```

---

### `plot_categorical_distributions(df, cat_vars, figsize=(5, 5))`
**Purpose**: Visualize distribution of categorical variables

**Inputs**:
- `df` (DataFrame): Input dataframe
- `cat_vars` (list): Categorical columns to plot
- `figsize` (tuple, default=(5, 5)): Figure size

**Outputs**:
- Displays: Pie chart for each categorical variable

**Example**:
```python
plot_categorical_distributions(df, ['GENDER', 'RACE', 'EDUCATION'])
```

---

## 6. Model Training

### `split_data(df, target_col='OUTCOME', test_size=0.2, val_size=0.1, random_state=42)`
**Purpose**: Split data into train/test/validation sets with stratification

**Inputs**:
- `df` (DataFrame): Input dataframe
- `target_col` (str, default='OUTCOME'): Target variable name
- `test_size` (float, default=0.2): Test set proportion (0-1)
- `val_size` (float, default=0.1): Validation set proportion (0-1)
- `random_state` (int, default=42): Random seed for reproducibility

**Outputs**:
- Returns: Tuple (X_train, X_test, X_val, y_train, y_test, y_val)

**Split Proportions** (default):
- Train: 70%
- Test: 20%
- Validation: 10%

**Example**:
```python
X_train, X_test, X_val, y_train, y_test, y_val = split_data(df)
# Output: Train: (6545, 18), Test: (1864, 18), Val: (932, 18)
```

---

### `train_logit(X_train, y_train, C=1.0, max_iter=1000, random_state=42)`
**Purpose**: Train Logistic Regression classifier

**Inputs**:
- `X_train` (array-like): Training features
- `y_train` (array-like): Training labels
- `C` (float, default=1.0): Inverse regularization strength
- `max_iter` (int, default=1000): Maximum iterations
- `random_state` (int, default=42): Random seed

**Outputs**:
- Returns: Trained LogisticRegression model

**Example**:
```python
logit_model = train_logit(X_train, y_train, C=0.5)
```

---

### `train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)`
**Purpose**: Train Random Forest classifier

**Inputs**:
- `X_train` (array-like): Training features
- `y_train` (array-like): Training labels
- `n_estimators` (int, default=100): Number of trees
- `max_depth` (int, default=10): Maximum tree depth
- `random_state` (int, default=42): Random seed
- `n_jobs` (int, default=-1): CPU cores to use (-1 = all)

**Outputs**:
- Returns: Trained RandomForestClassifier model

**Example**:
```python
rf_model = train_random_forest(X_train, y_train, n_estimators=200, max_depth=15)
```

---

### `train_xgboost(X_train, y_train, **kwargs)`
**Purpose**: Train XGBoost classifier with automatic class balancing

**Inputs**:
- `X_train` (array-like): Training features
- `y_train` (array-like): Training labels
- `**kwargs`: Additional XGBoost parameters

**Default Parameters**:
```python
{
    'n_estimators': 100,
    'scale_pos_weight': auto_calculated,  # Handles class imbalance
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'logloss'
}
```

**Outputs**:
- Returns: Trained XGBClassifier model

**Example**:
```python
xgb_model = train_xgboost(X_train, y_train, max_depth=8, learning_rate=0.05)
```

---

## 7. Model Evaluation

### `evaluate_model(model, X_test, y_test, model_name="Model", is_xgb=False, threshold=0.5)`
**Purpose**: Comprehensive model evaluation with standard metrics

**Inputs**:
- `model`: Trained model object
- `X_test` (array-like): Test features
- `y_test` (array-like): Test labels
- `model_name` (str, default="Model"): Name for display
- `is_xgb` (bool, default=False): Whether model is XGBoost (affects prediction method)
- `threshold` (float, default=0.5): Classification threshold

**Outputs**:
- Prints: Performance metrics
- Returns: Dictionary with metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC

**Example**:
```python
metrics = evaluate_model(xgb_model, X_test, y_test, model_name="XGBoost", is_xgb=True, threshold=0.549)
# Output:
# === XGBoost PERFORMANCE ===
# Threshold   : 0.5490
# Accuracy    : 0.8519
# Precision   : 0.6813
# Recall      : 0.8583
# F1          : 0.7595
# ROC-AUC     : 0.9164
```

---

### `compare_models(metrics_list)`
**Purpose**: Compare multiple models side-by-side

**Inputs**:
- `metrics_list` (list): List of dictionaries from `evaluate_model()`

**Outputs**:
- Displays: Comparison table and bar chart
- Returns: Comparison DataFrame

**Example**:
```python
metrics_list = [
    evaluate_model(logit_model, X_test, y_test, "Logistic"),
    evaluate_model(rf_model, X_test, y_test, "RandomForest"),
    evaluate_model(xgb_model, X_test, y_test, "XGBoost", is_xgb=True)
]
comparison = compare_models(metrics_list)
```

---

## 8. Hyperparameter Optimization

### `optimize_xgboost_optuna(X_train, y_train, X_val=None, y_val=None, n_trials=100, ...)`
**Purpose**: Automated hyperparameter tuning with Optuna (Bayesian optimization)

**Inputs**:
- `X_train`, `y_train`: Training data
- `X_val`, `y_val` (optional): Validation data (required if use_cv=False)
- `n_trials` (int, default=100): Number of optimization trials
- `direction` (str, default='maximize'): Optimization direction
- `cv_folds` (int, default=10): Cross-validation folds
- `use_cv` (bool, default=True): Use CV instead of validation set
- `random_state` (int, default=42): Random seed
- `n_jobs` (int, default=-1): Parallel jobs
- `overfitting_penalty` (float, default=0.1): Penalty for train-test gap

**Search Space**:
- n_estimators: 100-500
- learning_rate: 0.005-0.3 (log scale)
- max_depth: 2-12
- min_child_weight: 1-20
- subsample: 0.5-1.0
- colsample_bytree: 0.5-1.0
- gamma: 0.01-15
- reg_alpha: 0.01-1
- reg_lambda: 1-10
- threshold: 0.1-0.9

**Outputs**:
- Prints: Best parameters, overfitting diagnostics, performance metrics
- Returns: Tuple (best_params_dict, best_threshold, study_object)

**Example**:
```python
best_params, best_threshold, study = optimize_xgboost_optuna(
    X_train, y_train,
    n_trials=100,
    use_cv=True,
    overfitting_penalty=0.1
)

# Train final model with best parameters
final_model = train_xgboost(X_train, y_train, **best_params)
```

**Output Example**:
```
====================================================================
BEST XGBOOST PARAMETERS (OPTUNA - F1 OPTIMIZED)
====================================================================
Best F1 Score: 0.7595

====================================================================
OVERFITTING DIAGNOSTICS
====================================================================
F1 Score (Train): 0.9383
F1 Score (CV): 0.7595
Precision (Train): 0.9762
Precision (CV): 0.6813
Recall (Train): 0.9043
Recall (CV): 0.8583

Overfitting Gap: 0.1788
⚠️ MODERATE - Some overfitting detected

====================================================================
BEST HYPERPARAMETERS
====================================================================
 n_estimators: 322
 learning_rate: 0.0891
 max_depth: 6
 min_child_weight: 13
 subsample: 0.8213
 colsample_bytree: 0.6897
 gamma: 2.4512
 reg_alpha: 0.4567
 reg_lambda: 7.8923

Best Threshold: 0.5490
```

---

## 9. Model Diagnostics

### `plot_xgboost_diagnostics(model, X_train, y_train, X_test, y_test, model_name="XGBoost")`
**Purpose**: Comprehensive visualization of model performance (6 plots)

**Inputs**:
- `model`: Trained XGBoost model
- `X_train`, `y_train`: Training data
- `X_test`, `y_test`: Test data
- `model_name` (str, default="XGBoost"): Name for plot titles

**Outputs**:
- Displays: 6-panel diagnostic plot:
  1. **ROC Curve**: True Positive Rate vs False Positive Rate
  2. **Precision-Recall Curve**: PR curve with optimal F1 threshold
  3. **Confusion Matrix**: TP, FP, TN, FN counts
  4. **Feature Importance**: Top 20 most important features
  5. **Calibration Curve**: Predicted vs actual probabilities
  6. **Prediction Distribution**: Histogram of predictions by class

- Prints: 5-section diagnostic summary:
  1. Classification metrics (precision, recall, F1)
  2. Probability metrics (ROC-AUC, Brier score, log loss)
  3. Optimal thresholds (ROC-based, PR-based)
  4. Class distribution (train vs test)
  5. Overfitting check (train vs test AUC)

- Returns: Dictionary with key metrics

**Example**:
```python
diagnostics = plot_xgboost_diagnostics(
    xgb_model, X_train, y_train, X_test, y_test
)
# diagnostics = {
#     'roc_auc': 0.9164,
#     'avg_precision': 0.8234,
#     'brier_score': 0.1051,
#     'optimal_threshold_roc': 0.4821,
#     'optimal_threshold_pr': 0.5490,
#     'confusion_matrix': array([[1450, 102], [52, 310]])
# }
```

---

### `plot_learning_curves(model, X_train, y_train, cv=5, scoring='roc_auc')`
**Purpose**: Diagnose bias-variance tradeoff (overfitting vs underfitting)

**Inputs**:
- `model`: Trained model (supports calibrated wrappers)
- `X_train`, `y_train`: Training data
- `cv` (int, default=5): Cross-validation folds
- `scoring` (str, default='roc_auc'): Metric to plot ('roc_auc', 'precision', 'recall', 'f1')

**Outputs**:
- Displays: Learning curve plot with train/validation scores vs dataset size
- Prints: Comprehensive diagnosis:
  - Final train/CV scores
  - Gap between train and CV
  - Overfitting/underfitting diagnosis
  - Recommendations for improvement

- Returns: Dictionary with diagnostics:
  - final_gap
  - train_auc
  - cv_auc
  - cv_std

**Interpretation**:
- **Large gap (>0.10)**: HIGH VARIANCE (Overfitting)
  - Solutions: Increase regularization, reduce complexity, more data
- **Both curves low (<0.80)**: HIGH BIAS (Underfitting)
  - Solutions: Increase complexity, reduce regularization, add features
- **Small gap (<0.05) & high CV (>0.85)**: PERFECT ✅

**Example**:
```python
learning_diagnostics = plot_learning_curves(xgb_model, X_train, y_train, cv=10)
# Output:
# 🧠 LEARNING CURVE DIAGNOSIS
# ====================================================================
# Final Train AUC: 0.9423
# Final CV AUC: 0.9380
# Gap: 0.0043
# CV Std: 0.0121
# 
# ✅ PERFECT - Optimal bias-variance tradeoff!
#    Ready for production
# 
# CV Stability: ✅ EXCELLENT (low variance)
```

---

### `plot_threshold_analysis(model, X_test, y_test)`
**Purpose**: Find optimal classification threshold

**Inputs**:
- `model`: Trained model
- `X_test`, `y_test`: Test data

**Outputs**:
- Displays: 2 plots:
  1. Threshold vs Metrics (Precision, Recall, F1, Accuracy)
  2. Threshold vs Error Rates (FPR, FNR)

- Prints: Threshold recommendations:
  - Best F1 threshold
  - Equal error rate threshold
  - High precision threshold (0.9+)
  - High recall threshold (0.9+)

**Example**:
```python
plot_threshold_analysis(xgb_model, X_test, y_test)
# Output:
# THRESHOLD RECOMMENDATIONS
# ====================================================================
# Best F1 threshold: 0.5490 (F1=0.7595)
# Equal error rate: 0.4821 (FPR=FNR=0.1234)
# High precision (0.9+): 0.7232
# High recall (0.9+): 0.3156
```

---

### `plot_feature_importance(model, feature_names, max_num_features=15, figsize=(10, 8))`
**Purpose**: Visualize XGBoost built-in feature importance

**Inputs**:
- `model`: Trained XGBoost model
- `feature_names` (list): Feature names
- `max_num_features` (int, default=15): Number of features to display
- `figsize` (tuple, default=(10, 8)): Figure size

**Outputs**:
- Displays: Horizontal bar chart of feature importances

**Example**:
```python
plot_feature_importance(xgb_model, X_train.columns, max_num_features=20)
```

---

## 10. Calibration

### `diagnose_calibration(model, X_train, y_train, X_test, y_test)`
**Purpose**: Check if predicted probabilities match actual outcomes

**Inputs**:
- `model`: Trained model
- `X_train`, `y_train`: Training data
- `X_test`, `y_test`: Test data

**Outputs**:
- Prints: Calibration metrics (Brier score, Log loss) for train and test
- Returns: Tuple (y_train_proba, y_test_proba)

**Interpretation**:
- Brier score: 0 (perfect) to 1 (worst)
- Good calibration: Brier < 0.10
- Log loss: Lower is better

**Example**:
```python
y_train_proba, y_test_proba = diagnose_calibration(
    xgb_model, X_train, y_train, X_test, y_test
)
# Output:
# CALIBRATION DIAGNOSTICS
# ==================================================
# Brier Score - Train: 0.0574, Test: 0.1051
# Log Loss - Train: 0.1983, Test: 0.3412
# Gap (Train-Test): Brier=0.0477, LogLoss=0.1429
```

---

### `fix_calibration_underestimation(model, X_train, y_train, X_test, y_test, method='isotonic')`
**Purpose**: Fix probability calibration issues (especially underestimation)

**Inputs**:
- `model`: Trained model
- `X_train`, `y_train`: Training data
- `X_test`, `y_test`: Test data
- `method` (str): Calibration method
  - `'isotonic'`: Non-parametric (best for underestimation)
  - `'platt'`: Sigmoid fit (Platt scaling)
  - `'beta'`: Beta calibration (robust)
  - `'shift'`: Simple probability shift

**Outputs**:
- Displays: Before/after calibration curves
- Prints: Brier score improvement
- Returns: Calibrated probabilities (array)

**Example**:
```python
y_calib = fix_calibration_underestimation(
    xgb_model, X_train, y_train, X_test, y_test, method='isotonic'
)
# Output:
# CALIBRATION FIX: ISOTONIC
# ==================================================
# Brier Score Improvement: 0.1051 → 0.0908 (13.6% better)
```

---

### `complete_calibration_pipeline(model, X_train, y_train, X_test, y_test)`
**Purpose**: Automated calibration pipeline - tries all methods and selects best

**Inputs**:
- `model`: Trained model
- `X_train`, `y_train`: Training data
- `X_test`, `y_test`: Test data

**Outputs**:
- Prints: Results for all methods, best method selection
- Returns: Tuple (final_calibrator, calibrated_probabilities, results_dict)

**Example**:
```python
calibrator, y_calib, results = complete_calibration_pipeline(
    xgb_model, X_train, y_train, X_test, y_test
)
# Output:
# ==================================================
# BEST CALIBRATION METHOD:
# ISOTONIC: Brier = 0.0908
```

---

## 11. Post-Hoc Analysis

### `analyze_prediction_errors(model, X_train, X_test, y_test, feature_names=None, threshold=0.549)`
**Purpose**: Deep dive into false positives and false negatives

**Inputs**:
- `model`: Trained model
- `X_train`: Training features (for reference)
- `X_test`: Test features
- `y_test`: Test labels
- `feature_names` (list, optional): Feature names (auto-detected if DataFrame)
- `threshold` (float, default=0.549): Classification threshold

**Outputs**:
- Prints: 4 detailed analyses:
  1. Error counts (FP, FN, TP, TN)
  2. False Positive feature statistics
  3. False Negative feature statistics
  4. FP vs TP comparison (why misclassified?)

- Displays: 2 figure sets:
  1. 6 histograms: Feature distributions by error type (TN, FP, TP, FN)
  2. 2 histograms: Probability distributions for FP and FN

- Returns: Dictionary with DataFrames:
  - fp_features: False positive samples
  - fn_features: False negative samples
  - tp_features: True positive samples
  - tn_features: True negative samples
  - comparison: FP vs TP feature comparison

**Example**:
```python
error_analysis = analyze_prediction_errors(
    xgb_model, X_train, X_test, y_test, threshold=0.549
)
# Output:
# ERROR ANALYSIS: FALSE POSITIVES & FALSE NEGATIVES
# ========================================================================
# False Positives: 102 (5.5%)
# False Negatives: 36 (1.9%)
# True Positives:  310 (16.6%)
# True Negatives:  1416 (76.0%)
```

---

### `analyze_shap_values(model, X_train, X_test, y_test)`
**Purpose**: Explainability analysis using SHAP (SHapley Additive exPlanations)

**Inputs**:
- `model`: Trained XGBoost/tree-based model
- `X_train`: Training features
- `X_test`: Test features
- `y_test`: Test labels

**Outputs**:
- Prints: 4-step analysis
- Displays: 4 visualizations:
  1. **SHAP Feature Importance Bar Chart**: Global importance (mean |SHAP|)
  2. **SHAP Beeswarm Plot**: Feature effects by value (red=high, blue=low)
  3. **Top 3 Dependence Plots**: Feature interactions
  4. **Force Plot**: Individual prediction explanation (false positive example)

- Returns: Tuple (explainer_object, shap_values_array)

**Example**:
```python
explainer, shap_values = analyze_shap_values(xgb_model, X_train, X_test, y_test)
# Takes 1-2 minutes to compute
```

---

### `compute_permutation_importance(model, X_test, y_test)`
**Purpose**: True predictive power by shuffling features (more reliable than built-in importance)

**Inputs**:
- `model`: Trained model
- `X_test`: Test features
- `y_test`: Test labels

**Outputs**:
- Prints: Top 15 features with importance scores
- Displays: Horizontal bar chart (top 20 features)
- Returns: DataFrame with columns (feature, importance, std)

**Interpretation**:
- Importance = AUC drop when feature is shuffled
- Higher importance = more predictive power

**Example**:
```python
perm_importance = compute_permutation_importance(xgb_model, X_test, y_test)
# Output:
# PERMUTATION IMPORTANCE (True Predictive Power)
# ====================================================================
# Computing (may take 30-60 seconds)...
# 
# Top 15 Most Important Features:
#             feature  importance      std
#   DRIVING_EXPERIENCE      0.0423  0.0089
#        CREDIT_SCORE      0.0312  0.0067
#                 AGE      0.0267  0.0054
```

---

### `plot_partial_dependence_fixed(model, X_test, feature_names=None, top_n=6)`
**Purpose**: Show marginal effect of each feature on predictions

**Inputs**:
- `model`: Trained model with feature_importances_ attribute
- `X_test`: Test features
- `feature_names` (list, optional): Feature names (auto-detected)
- `top_n` (int, default=6): Number of features to plot

**Outputs**:
- Displays: 2x3 grid of partial dependence plots
- Returns: List of top feature names

**Interpretation**:
- Shows how prediction changes as feature value varies
- Reveals non-linear relationships

**Example**:
```python
top_features = plot_partial_dependence_fixed(xgb_model, X_test)
# Output:
# Feature names detected: 18 features
# Top 6 features by XGBoost importance: ['DRIVING_EXPERIENCE', 'CREDIT_SCORE', ...]
# ✅ Partial Dependence plots generated successfully!
```

---

## 12. Business Analytics

### `business_impact_analysis(model, X_test, y_test, threshold=0.549)`
**Purpose**: Translate model performance into business metrics (cost-benefit, risk stratification, pricing)

**Inputs**:
- `model`: Trained model
- `X_test`: Test features
- `y_test`: Test labels
- `threshold` (float, default=0.549): Current classification threshold

**Outputs**:
- Prints: 3-part business analysis:
  1. **Cost-Benefit Analysis**: Optimal threshold for profit maximization
  2. **Risk Stratification**: Performance by risk tier
  3. **Premium Pricing Simulation**: Profitability analysis

- Displays: 3 visualizations:
  1. Net profit vs threshold curve
  2. Precision/Recall at optimal threshold
  3. Claim rate by risk tier (bar chart)

- Returns: Dictionary with:
  - optimal_threshold: Profit-maximizing threshold
  - max_profit: Maximum achievable profit
  - current_profit: Profit at current threshold
  - tier_analysis: DataFrame with risk tier statistics
  - loss_ratio: Total claims / total premiums

**Cost Assumptions** (customizable):
- FP cost: $150 (investigation cost)
- FN cost: $8,000 (missed claim)
- TP benefit: $5,000 (prevented claim)

**Example**:
```python
business_results = business_impact_analysis(xgb_model, X_test, y_test)
# Output:
# BUSINESS IMPACT ANALYSIS
# ====================================================================
# 
# 1. COST-BENEFIT ANALYSIS
# --------------------------------------------------------------------
# Cost per False Positive:  $150
# Cost per False Negative:  $8,000
# Benefit per True Positive: $5,000
# 
# Current Threshold (0.549):
#   Net Profit: $1,247,800.00
#   Precision: 0.681
#   Recall: 0.858
# 
# Optimal Business Threshold (0.620):
#   Net Profit: $1,356,200.00
#   Improvement: $108,400.00 (8.7%)
#   Precision: 0.734
#   Recall: 0.812
# 
# 2. RISK STRATIFICATION
# --------------------------------------------------------------------
# Risk_Tier    Count  Claims  Claim_Rate
# Very Low       373      28      0.075
# Low            373      63      0.169
# Medium         373     103      0.276
# High           372     134      0.360
# Very High      373     234      0.627
# 
# 3. PREMIUM PRICING SIMULATION
# --------------------------------------------------------------------
# Base Premium:      $1,200
# Avg Claim Amount:  $8,000
# 
# Total Premiums:    $3,128,456.78
# Total Claims Paid: $2,896,000.00
# Net Profit:        $232,456.78
# Loss Ratio:        92.57% (target: <75%)
# ⚠️  HIGH LOSS RATIO - Consider premium adjustments
```

---

### `explain_single_prediction(model, X_test, idx, threshold=0.549, top_features=10)`
**Purpose**: Explain why model made a specific prediction for individual customer

**Inputs**:
- `model`: Trained model
- `X_test`: Test features (DataFrame or array)
- `idx` (int): Index of sample to explain
- `threshold` (float, default=0.549): Classification threshold
- `top_features` (int, default=10): Number of features to display

**Outputs**:
- Prints: Prediction probability and top 10 features
- Displays: Horizontal bar chart of feature importances for this prediction
- Returns: DataFrame with top features for this sample

**Example**:
```python
sample_explanation = explain_single_prediction(xgb_model, X_test, idx=42, threshold=0.549)
# Output:
# Prediction 42: Proba=0.823 → CLAIM
# 
# Top 10 Features Driving This Decision:
#                  feature  importance  value
#     DRIVING_EXPERIENCE      0.1523     0.0
#           CREDIT_SCORE      0.1234   450.0
#                    AGE      0.0987     1.0
```

---

### `logistic_regression_diagnostics(logit_model, X_train, X_test, y_train, y_test, ...)`
**Purpose**: Specialized diagnostics for Logistic Regression (coefficients, residuals)

**Inputs**:
- `logit_model`: Trained LogisticRegression model
- `X_train`, `y_train`: Training data
- `X_test`, `y_test`: Test data
- `feature_names` (list, optional): Feature names
- `threshold` (float, default=0.5): Classification threshold

**Outputs**:
- Displays: 5-panel diagnostic plot:
  1. ROC Curve
  2. Precision-Recall Curve
  3. Confusion Matrix
  4. Calibration Curve
  5. **Logistic Coefficients** (unique to logistic regression)

- Prints: Metrics table and coefficient summary
- Returns: Tuple (metrics_dict, coefficients_dataframe)

**Example**:
```python
metrics, coefs = logistic_regression_diagnostics(
    logit_model, X_train, X_test, y_train, y_test
)
# Coefficient interpretation:
# - Positive coefficient → increases claim probability
# - Negative coefficient → decreases claim probability
# - Magnitude shows strength of effect
```

---

### `logistic_residual_diagnostics_fixed(logit_model, X_train, X_test, y_train, y_test, threshold=0.549)`
**Purpose**: Residual analysis for logistic regression (check model assumptions)

**Inputs**:
- `logit_model`: Trained LogisticRegression model
- `X_train`, `y_train`: Training data
- `X_test`, `y_test`: Test data
- `threshold` (float, default=0.549): Classification threshold

**Outputs**:
- Displays: 5 diagnostic plots:
  1. **Residuals vs Fitted**: Check for systematic bias (should be random scatter)
  2. **Q-Q Plot**: Check normality of residuals
  3. **Residuals Histogram**: Distribution of residuals (should center at 0)
  4. **Scale-Location**: Check heteroscedasticity (constant variance)
  5. **Residuals vs Order**: Check for autocorrelation

- Prints: Residual statistics and interpretation
- Returns: Dictionary with residuals and statistics

**Interpretation**:
- ✅ **Good**: Residuals random, centered at 0, normal distribution
- ⚠️ **Warning**: Patterns indicate model assumptions violated

**Example**:
```python
residual_diagnostics = logistic_residual_diagnostics_fixed(
    logit_model, X_train, X_test, y_train, y_test
)
# Output:
# RESIDUAL DIAGNOSTICS SUMMARY
# ====================================================================
# Mean Residual                -0.0023
# Median Residual              -0.0156
# Std Residuals                 0.3912
# Max |Residual|                0.9823
# Skewness                      0.1234
# Kurtosis                      2.8765
# KS p-value                    0.0871
# Train AUC                     0.8123
# Test AUC                      0.7965
# 
# INTERPRETATION:
# ----------------------------------------
# ✅ MEAN ≈ 0: No systematic bias
# ✅ RESIDUALS NORMAL: Good for inference
```

---

## 📊 Typical Analysis Workflow

```python
# 1. Load and explore
df, df_backup = load_data('insurance.csv')
display_basic_info(df)

# 2. Handle missing data
missing_report = analyze_missing_data(df)
mechanism = analyze_missingness_mechanism(df)
df_clean = drop_missing_values(df)

# 3. Encode features
df_clean = encode_ordinal_features(df_clean)
df_clean = encode_binary_features(df_clean)
df_clean = optimize_numeric_dtypes(df_clean)

# 4. Statistical analysis
normality = test_normality(df_clean)
correlations = plot_correlation_heatmap(df_clean)

# 5. Split data
X_train, X_test, X_val, y_train, y_test, y_val = split_data(df_clean)

# 6. Train models
logit = train_logit(X_train, y_train)
xgb = train_xgboost(X_train, y_train)

# 7. Optimize (optional but recommended)
best_params, best_threshold, study = optimize_xgboost_optuna(
    X_train, y_train, n_trials=100
)
xgb_optimized = train_xgboost(X_train, y_train, **best_params)

# 8. Evaluate
metrics_xgb = evaluate_model(xgb_optimized, X_test, y_test, "XGBoost", is_xgb=True, threshold=best_threshold)
diagnostics = plot_xgboost_diagnostics(xgb_optimized, X_train, y_train, X_test, y_test)

# 9. Learning curves
learning_results = plot_learning_curves(xgb_optimized, X_train, y_train)

# 10. Calibration
calibrator, y_calib, calib_results = complete_calibration_pipeline(
    xgb_optimized, X_train, y_train, X_test, y_test
)

# 11. Post-hoc analysis
error_analysis = analyze_prediction_errors(xgb_optimized, X_train, X_test, y_test, threshold=best_threshold)
explainer, shap_values = analyze_shap_values(xgb_optimized, X_train, X_test, y_test)
perm_importance = compute_permutation_importance(xgb_optimized, X_test, y_test)

# 12. Business metrics
business_results = business_impact_analysis(xgb_optimized, X_test, y_test, threshold=best_threshold)
```

---

## 🎯 Key Performance Indicators

| Metric | Good | Excellent | Notes |
|--------|------|-----------|-------|
| ROC-AUC | > 0.85 | > 0.92 | Overall discrimination ability |
| Precision | > 0.65 | > 0.75 | % of flagged claims that are real |
| Recall | > 0.80 | > 0.90 | % of real claims caught |
| F1 Score | > 0.70 | > 0.80 | Harmonic mean of precision/recall |
| Brier Score | < 0.12 | < 0.09 | Calibration quality (lower better) |
| Learning Curve Gap | < 0.10 | < 0.05 | Overfitting check |
| Loss Ratio | < 75% | < 65% | Claims paid / premiums collected |

---

## 📦 Dependencies

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import xgboost as xgb
from xgboost import XGBClassifier
import shap
import optuna
```

---

## 📝 Citation

If you use these functions, please cite:

```
Car Insurance Risk Analysis Framework
Author: [Your Name]
Year: 2026
Description: Comprehensive ML pipeline for insurance claim prediction
```

---

## ⚠️ Important Notes

1. **Class Imbalance**: `train_xgboost()` automatically handles imbalance using `scale_pos_weight`
2. **Thresholds**: Default threshold (0.5) is often suboptimal - always run `plot_threshold_analysis()`
3. **Calibration**: XGBoost often underestimates probabilities - use `complete_calibration_pipeline()`
4. **Overfitting**: Always check `plot_learning_curves()` before production
5. **Business Metrics**: Adjust cost parameters in `business_impact_analysis()` to match your domain

---

## 🐛 Troubleshooting

**KeyError with partial_dependence**: Use `plot_partial_dependence_fixed()` instead

**SHAP too slow**: Reduce test set size or sample: `X_test.sample(1000)`

**Optuna convergence issues**: Increase `n_trials` to 200+

**Memory errors**: Use `optimize_numeric_dtypes()` before modeling

---

## 📧 Contact

For questions or issues with these functions, please open an issue on the repository.

---

**Last Updated**: February 2026  
**Version**: 1.0
