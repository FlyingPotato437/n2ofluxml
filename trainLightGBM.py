import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt

data = pd.read_csv('puturpathere')

numeric_columns = ['N2O', 'N_rate', 'PP2', 'PP7', 'AirT', 'DAF_TD', 'DAF_SD', 'WFPS25cm', 'NH4', 'NO3', 'Clay', 'Sand', 'SOM']
data = data[numeric_columns + ['Vegetation', 'Experiment']]

imputer = IterativeImputer(random_state=500)
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

data['bins'] = pd.qcut(data['N2O'], q=5, labels=['verylow', 'low', 'mediumlow', 'medium', 'high'])

data = data.dropna(subset=['bins'])

def print_bin_counts(data, group_col):
    for group, group_data in data.groupby(group_col):
        print(f"Group: {group}")
        print(group_data['bins'].value_counts())
        print()

print("Bin counts for Experiment:")
print_bin_counts(data, 'Experiment')

print("Bin counts for Vegetation:")
print_bin_counts(data, 'Vegetation')

def train_and_evaluate(data, group_col):
    results = {}
    for group, group_data in data.groupby(group_col):
        print(f"Training model for {group_col}: {group}")

        y_binned_counts = group_data['bins'].value_counts()
        if y_binned_counts.min() < 2:
            print(f"Skipping {group_col}: {group} due to insufficient class samples in bins: {y_binned_counts.to_dict()}")
            continue

        X = group_data[numeric_columns]
        y = group_data['N2O']
        y_binned = group_data['bins']

        X_train, X_eval, y_train, y_eval, y_train_binned, y_eval_binned = train_test_split(
            X, y, y_binned, test_size=0.3, stratify=y_binned, random_state=500
        )

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_eval_scaled = scaler.transform(X_eval)

        param_grid = {
            'num_leaves': [31, 50, 100],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 300, 500]
        }

        model = lgb.LGBMRegressor(random_state=500)
        kfold = KFold(n_splits=5, shuffle=True, random_state=500)
        grid_search = GridSearchCV(model, param_grid, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)

        best_model = grid_search.best_estimator_

        y_train_pred = best_model.predict(X_train_scaled)
        y_eval_pred = best_model.predict(X_eval_scaled)

        train_mse = mean_squared_error(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        eval_mse = mean_squared_error(y_eval, y_eval_pred)
        eval_mae = mean_absolute_error(y_eval, y_eval_pred)
        eval_r2 = r2_score(y_eval, y_eval_pred)

        results[group] = {
            'train_mse': train_mse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'eval_mse': eval_mse,
            'eval_mae': eval_mae,
            'eval_r2': eval_r2,
            'model': best_model
        }

        plt.figure(figsize=(10, 5))
        plt.plot(y_eval.values, label='True')
        plt.plot(y_eval_pred, label='Predicted')
        plt.title(f'{group_col}: {group} - Evaluation')
        plt.xlabel('Sample Index')
        plt.ylabel('N2O Flux')
        plt.legend()
        plt.show()

        print(f'Results for {group_col}: {group}')
        print(f'Train MSE: {train_mse}, Train MAE: {train_mae}, Train R^2: {train_r2}')
        print(f'Eval MSE: {eval_mse}, Eval MAE: {eval_mae}, Eval R^2: {eval_r2}')
        print('-' * 60)

    return results

experiment_results = train_and_evaluate(data, 'Experiment')
vegetation_results = train_and_evaluate(data, 'Vegetation')

print("Experiment Results:")
for experiment, result in experiment_results.items():
    print(f'{experiment}: {result}')

print("\nVegetation Results:")
for vegetation, result in vegetation_results.items():
    print(f'{vegetation}: {result}')
