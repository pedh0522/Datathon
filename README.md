# Diabetes Classification using Machine Learning and Neural Networks

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

1. **Model Performance**:
   - Accuracy, ROC AUC, and classification report for each model.
   
2. **Best Model**:
   - The best model based on ROC-AUC score after hyperparameter tuning.

3. **Ensemble Results**:
   - Accuracy and ROC-AUC score for the ensemble method (Stacking or Voting).

4. **Neural Network Results**:
   - Accuracy, ROC-AUC, and classification report for the neural network model.

## Conclusion

This project combines multiple approaches to predict diabetes using machine learning and neural networks. It provides flexibility in terms of the techniques used, such as oversampling, hyperparameter tuning, feature selection, and ensembling.

## Acknowledgements

- **Optuna** for hyperparameter optimization.
- **XGBoost** and **CatBoost** for powerful gradient boosting models.
- **PyTorch** for implementing the neural network model.
