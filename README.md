# Diabetes Classification

This project aims to predict whether a patient has diabetes using various machine learning models, data preprocessing techniques, and a neural network classifier. It utilizes multiple classification algorithms like Logistic Regression, Decision Trees, Random Forest, XGBoost, CatBoost, and a Neural Network (PyTorch). Additionally, techniques such as PCA (Principal Component Analysis), feature selection, oversampling (SMOTE and ADASYN), and hyperparameter tuning via GridSearch, RandomSearch, and Bayesian optimization are employed.

## Requirements

1. **Python 3.x**
2. **Libraries**:
    - `pandas`
    - `numpy`
    - `sklearn`
    - `xgboost`
    - `catboost`
    - `torch`
    - `imblearn`
    - `optuna`
    - `argparse`
   
To install the required libraries, run:

```bash
pip install -r requirements.txt
```

## Project Overview

### 1. **Data Preprocessing**
The project utilizes multiple datasets, including a diabetes dataset (SAS format), and other health-related datasets (e.g., smoking, alcohol use, blood pressure, and kidney conditions). The preprocessing steps include:

- Merging datasets.
- Handling missing values using KNN imputation or Hot Deck imputation.
- Adding diabetes status based on predefined columns.
- Dropping columns with too many missing values.
- Normalizing and standardizing data.

### 2. **Feature Engineering**
Optional features include:

- **Principal Component Analysis (PCA)**: Reduces dimensionality of the dataset.
- **Feature Selection**: Identifies important features using RandomForest.

### 3. **Oversampling for Imbalanced Classes**
- **SMOTE**: Synthetic Minority Over-sampling Technique.
- **ADASYN**: Adaptive Synthetic Sampling.

### 4. **Machine Learning Models**
The following models are used for classification:

- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **XGBoost**
- **CatBoost**

### 5. **Hyperparameter Tuning**
Hyperparameter optimization is done using:

- **GridSearchCV**: Exhaustive search over a specified parameter grid.
- **RandomizedSearchCV**: Random search over a parameter grid.
- **Optuna (Bayesian Optimization)**: Efficient hyperparameter optimization.

### 6. **Ensemble Methods**
- **Stacking Classifier**: Combines multiple models' predictions.
- **Voting Classifier**: Uses soft voting based on probability predictions.

### 7. **Neural Network Model**
A neural network model is built using **PyTorch** with:

- A **simple feedforward architecture**.
- **CrossEntropyLoss** for multi-class classification.
- **Adam optimizer**.

### 8. **Metrics**
The following metrics are used for evaluation:

- **Accuracy**
- **ROC-AUC**
- **Classification Report** (Precision, Recall, F1 Score)

## Usage

To run the script and train models with various configurations, use the following command:

```bash
python main.py [OPTIONS]
```

### Options:

- `--knn_imputation`: Apply KNN imputation to handle missing values.
- `--select_feat`: Select top features using RandomForest feature importance.
- `--pca`: Perform PCA to reduce the number of features.
- `--class_weight`: Apply class weight balancing in classifiers.
- `--baysian_opt`: Enable Bayesian Optimization for hyperparameter tuning.
- `--stack_ensemble`: Use stacking ensemble learning method.
- `--smote`: Apply SMOTE for oversampling.
- `--adasyn`: Apply ADASYN for oversampling.

### Example:

```bash
python main.py --knn_imputation --pca --baysian_opt --stack_ensemble
```

This command will perform KNN imputation, apply PCA, optimize hyperparameters using Optuna, and use a stacking ensemble for classification.

## Results

After running the script, the following outputs will be printed:

Model: Logistic Regression
Accuracy: 0.8516595380667237
ROC AUC: 0.7541385086351297
Classification Report:
              precision    recall  f1-score   support

           0       0.30      0.57      0.40       174
           1       0.91      0.77      0.84       995

    accuracy                           0.74      1169
   macro avg       0.61      0.67      0.62      1169
weighted avg       0.82      0.74      0.77      1169

--------------------------------------------------
Model: Decision Tree
Accuracy: 0.8041060735671514
ROC AUC: 0.7775140068156876
Classification Report:
              precision    recall  f1-score   support

           0       0.37      0.44      0.40       174
           1       0.90      0.87      0.88       995

    accuracy                           0.80      1169
   macro avg       0.63      0.65      0.64      1169
weighted avg       0.82      0.80      0.81      1169

--------------------------------------------------
Model: Random Forest
Accuracy: 0.8494439692044482
ROC AUC: 0.7990354069196559
Classification Report:
              precision    recall  f1-score   support

           0       0.49      0.21      0.30       174
           1       0.87      0.96      0.92       995

    accuracy                           0.85      1169
   macro avg       0.68      0.59      0.61      1169
weighted avg       0.82      0.85      0.82      1169

--------------------------------------------------
Model: XGBoost
Accuracy: 0.8374679213002566
ROC AUC: 0.7742909952059146
Classification Report:
              precision    recall  f1-score   support

           0       0.42      0.24      0.30       174
           1       0.88      0.94      0.91       995

    accuracy                           0.84      1169
   macro avg       0.65      0.59      0.60      1169
weighted avg       0.81      0.84      0.82      1169

--------------------------------------------------
Model: CatBoost
Accuracy: 0.8485885372112917
ROC AUC: 0.798596430428002
Classification Report:
              precision    recall  f1-score   support

           0       0.48      0.25      0.33       174
           1       0.88      0.95      0.91       995

    accuracy                           0.85      1169
   macro avg       0.68      0.60      0.62      1169
weighted avg       0.82      0.85      0.83      1169

--------------------------------------------------

--- Ensemble Model Results ---
Accuracy: 0.8417450812660393
ROC AUC: 0.7970715647201524
Accuracy: 0.8100940975192472
ROC AUC: 0.7164904984693583
Classification Report:
              precision    recall  f1-score   support

           0       0.32      0.25      0.28       174
           1       0.87      0.91      0.89       995

    accuracy                           0.81      1169
   macro avg       0.60      0.58      0.59      1169
weighted avg       0.79      0.81      0.80      1169

## Conclusion

This project combines multiple approaches to predict diabetes using machine learning and neural networks. It provides flexibility in terms of the techniques used, such as oversampling, hyperparameter tuning, feature selection, and ensembling.

## Acknowledgements

- **Optuna** for hyperparameter optimization.
- **XGBoost** and **CatBoost** for powerful gradient boosting models.
- **PyTorch** for implementing the neural network model.
