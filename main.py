import pandas as pd
import os
import numpy as np
import optuna
from preprocessor import Preprocessor
from feature_analysis import DimReductAndFeatureSelect
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import VotingClassifier, StackingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from torch.utils.data import DataLoader, TensorDataset
from neural_net import DiabetesClassifier
import torch
from torch import nn, optim

def main(args):
    path = 'D:\\Downloads\\raw_datasets\\'

    with open("D:\\Downloads\\variables.csv", mode='r', encoding='utf-8', errors='replace') as f:
        df_vars = pd.read_csv(f)

    diabetes = pd.read_sas(path + 'diabetes.XPT', format='xport')

    relevant_rows = df_vars[(df_vars['Variable Name'].isin(diabetes.columns)) & (df_vars['Data File Name'] == 'DIQ_L')]
    df_dia = diabetes.rename(columns=dict(zip(relevant_rows['Variable Name'], relevant_rows['Renamed_variables'])))

    preprocessor = Preprocessor(df_vars)

    file_info = [(path + 'alcohol_use.XPT', 'ALQ_L'), (path + 'smoking_cigarette_use.XPT', 'SMQ_L'), (path + 'blood_pressure_cholesterol.XPT', 'BPQ_L'), (path + 'kidney_condition_urology.XPT', 'KIQ_U_L')]
    merged_dataframe = preprocessor.process_xpt_files(file_info)

    merged_dataframe = preprocessor.drop_nan_columns(merged_dataframe, threshold=len(merged_dataframe))

    if args.knn_imputation:
        merged_dataframe = preprocessor.impute_missing_values(merged_dataframe)
        df_dia = preprocessor.impute_missing_values(df_dia)
    else:
        merged_dataframe = preprocessor.hot_deck_imputation(merged_dataframe)
        df_dia = preprocessor.hot_deck_imputation(df_dia)
 
    data = preprocessor.add_diabetes_status(merged_dataframe, df_dia, 'EverTold_Diabetes')

    valid_values = [1.0, 2.0]
    data = data[data['EverTold_Diabetes'].isin(valid_values)]

    data['EverTold_Diabetes'] = data['EverTold_Diabetes'].map({1.0: 0, 2.0: 1})

    X = data.drop(columns=['EverTold_Diabetes'])
    y = data['EverTold_Diabetes']
    print(y.isna().sum())
    if args.pca:
        feature_selector = DimReductAndFeatureSelect(X, y)
        X = feature_selector.perform_pca(X, n_components=2)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if args.smote:
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    
    if args.adasyn:
        adasyn = ADASYN(sampling_strategy='auto', random_state=42)
        X_train, y_train = adasyn.fit_resample(X_train, y_train)


    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(y_train.value_counts())
    print(y_test.value_counts())
    print(len(data.columns))

    if args.select_feat:
        feature_selector = DimReductAndFeatureSelect(X_train, y_train, X_test)
        feature_selector.select_top_features()
        X_train = feature_selector.X_train_selected
        X_test = feature_selector.X_test_selected
    class_weight = 'balanced' if args.class_weight else None

    models = {
    "Logistic Regression": LogisticRegression(class_weight=class_weight),
    "Decision Tree": DecisionTreeClassifier(class_weight=class_weight),
    "Random Forest": RandomForestClassifier(class_weight=class_weight),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'), 
    "CatBoost": CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, verbose=100)
    }
    if args.baysian_opt:
        def objective(trial):
            model_name = trial.suggest_categorical("model", ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "CatBoost"])
            
            if model_name == "Logistic Regression":
                params = {
                    'C': trial.suggest_loguniform('C', 0.01, 10),
                    'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs'])
                }
                model = LogisticRegression(class_weight='balanced', **params)
            
            elif model_name == "Decision Tree":
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }
                model = DecisionTreeClassifier(class_weight='balanced', **params)
            
            elif model_name == "Random Forest":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }
                model = RandomForestClassifier(class_weight='balanced', **params)
            
            elif model_name == "XGBoost":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_uniform('gamma', 0, 5)
                }
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **params)
            
            elif model_name == "CatBoost":
                params = {
                    'iterations': trial.suggest_int('iterations', 200, 500),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                    'depth': trial.suggest_int('depth', 4, 10)
                }
                model = CatBoostClassifier(verbose=0, **params)
            
            score = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=5, n_jobs=-1).mean()
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)

        best_params = study.best_params
        best_model_name = best_params.pop("model")
        print(f"Best Model: {best_model_name}")
        print(f"Best Parameters: {best_params}")

        # Mapping model names to their classes
        model_mapping = {
            "Logistic Regression": LogisticRegression,
            "Decision Tree": DecisionTreeClassifier,
            "Random Forest": RandomForestClassifier,
            "XGBoost": XGBClassifier,
            "CatBoost": CatBoostClassifier
        }

        # Train the best model with optimized parameters
        final_model = model_mapping[best_model_name](**best_params)
        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)
        y_proba = final_model.predict_proba(X_test)[:, 1] if hasattr(final_model, "predict_proba") else None

        accuracy = final_model.score(X_test, y_test)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        report = classification_report(y_test, y_pred)

        print(f"Final Model Accuracy: {accuracy}")
        print(f"Final Model ROC AUC: {roc_auc}")
        print(report)

        # Evaluate accuracy of all models
        for model_name in model_mapping.keys():
            model = model_mapping[model_name]()
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            print(f"{model_name} Accuracy: {acc}")


    else:
        param_grids = {
            "Logistic Regression": {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs']
            },
            "Decision Tree": {
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "Random Forest": {
                'n_estimators': [100, 200, 500],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "XGBoost": {
                'n_estimators': [100, 200, 500],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 10]
            },
            "CatBoost": {
                'iterations': [20, 50, 100],
                'learning_rate': [0.01, 0.05, 0.1],
                'depth': [4, 6, 10]
            }
        }

        # Train and evaluate models
        results = {}
        for name, model in models.items():
            print(f"Tuning {name}...")
            search = RandomizedSearchCV(model, param_grids[name], scoring='roc_auc', cv=5, n_iter=10, n_jobs=-1, verbose=1)
            search.fit(X_train, y_train)
            
            best_model = search.best_estimator_
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
            
            accuracy = best_model.score(X_test, y_test)
            roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
            report = classification_report(y_test, y_pred)
            
            results[name] = {
                "Best Params": search.best_params_,
                "Accuracy": accuracy,
                "ROC AUC": roc_auc,
                "Classification Report": report
            }


        # Print results
        for name, result in results.items():
            print(f"Model: {name}")
            print(f"Accuracy: {result['Accuracy']}")
            print(f"ROC AUC: {result['ROC AUC']}")
            print("Classification Report:")
            print(result['Classification Report'])
            print("-" * 50)
        

        # Ensemble

        best_models = {
            "Logistic Regression": LogisticRegression(**results["Logistic Regression"]["Best Params"]),
            "Decision Tree": DecisionTreeClassifier(**results["Decision Tree"]["Best Params"]),
            "Random Forest": RandomForestClassifier(**results["Random Forest"]["Best Params"]),
            "XGBoost": XGBClassifier(**results["XGBoost"]["Best Params"]),
            "CatBoost": CatBoostClassifier(**results["CatBoost"]["Best Params"], verbose=0)
        }

        if args.stack_ensemble:
            meta_model = LogisticRegression()  

            stacking_clf = StackingClassifier(
                estimators=[(name, model) for name, model in best_models.items()],
                final_estimator=meta_model,
                stack_method='auto' 
            )

            stacking_clf.fit(X_train, y_train)

            # Predict and Evaluate
            y_pred_stack = stacking_clf.predict(X_test)
            y_proba_stack = stacking_clf.predict_proba(X_test)[:, 1]

            stacking_accuracy = accuracy_score(y_test, y_pred_stack)
            stacking_roc_auc = roc_auc_score(y_test, y_proba_stack)

            print("\n--- Stacking Model Results ---")
            print(f"Accuracy: {stacking_accuracy}")
            print(f"ROC AUC: {stacking_roc_auc}")
            
        else:
            # Ensemble using Voting Classifier (soft voting for probability-based prediction)
            voting_clf = VotingClassifier(
                estimators=[(name, model) for name, model in best_models.items() if name != 'Logistic Regression'],
                voting='soft'  # Use 'hard' if models do not support predict_proba
            )

            # Train Voting Classifier
            voting_clf.fit(X_train, y_train)

            # Predict and Evaluate
            y_pred_ensemble = voting_clf.predict(X_test)
            y_proba_ensemble = voting_clf.predict_proba(X_test)[:, 1]

            ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
            ensemble_roc_auc = roc_auc_score(y_test, y_proba_ensemble)

            print("\n--- Ensemble Model Results ---")
            print(f"Accuracy: {ensemble_accuracy}")
            print(f"ROC AUC: {ensemble_roc_auc}")

            
        # Neural Network

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)  # Use .values for pandas series
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

        # Create DataLoader
        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        # Initialize the model
        input_size = X_train.shape[1]
        model = DiabetesClassifier(input_size=input_size)

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()  # For multi-class classification
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        epochs = 100
        best_roc_auc = 0

        for epoch in range(epochs):
            model.train()
            running_loss = 0
            
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model.eval()
        y_true = []
        y_pred = []
        y_proba = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                y_true.extend(y_batch.numpy())
                y_pred.extend(predicted.numpy())
                y_proba.extend(probabilities[:, 1].numpy())

        # Metrics
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        roc_auc = roc_auc_score(y_true, y_proba)
        report = classification_report(y_true, y_pred, zero_division=0)

        print("\n--- Neural Network Results ---")
        print(f"Accuracy: {accuracy}")
        print(f"ROC AUC: {roc_auc}")
        print("Classification Report:")
        print(report)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process health-related datasets")
    parser.add_argument('--knn_imputation', action='store_true', help="Apply KNN imputation to the data")
    parser.add_argument('--select_feat', action='store_true', help="Select top features using RandomForest")
    parser.add_argument('--pca', action='store_true', help="Perform PCA on the data")
    parser.add_argument('--class_weight', action='store_true', help="Apply weight balancing")
    parser.add_argument('--baysian_opt', action='store_true', help="Apply Bayesian optimization to hyperparameter tuning")
    parser.add_argument('--stack_ensemble', action='store_true', help="Apply stacking ensemble")
    parser.add_argument('--smote', action='store_true', help="Apply SMOTE for oversampling")
    parser.add_argument('--adasyn', action='store_true', help="Apply ADASYN for oversampling")
    args = parser.parse_args()
    main(args)