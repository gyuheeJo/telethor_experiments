import pandas as pd
import numpy as np
from preprocess import ts_preprocess, balance, scale
import optuna
import json
from train import lstm_torch_cv
import os

data_dirs = os.listdir("../data")
dfs = [pd.read_csv(f"../data/{dir}") for dir in data_dirs]
dataframe = pd.concat(dfs, ignore_index=True)
dataframe.head()

dataframe["datetime"] = pd.to_datetime(dataframe["datetime"])
dataframe_fs = dataframe.drop(['thunder_count', 'temp_min', 'temp_max', 'feels_like'], axis=1)

df_sorted = dataframe_fs.sort_values(by=["city_name", "datetime"])
groups = df_sorted.groupby('city_name')

dataset_ts = {}

for key, group in groups:

    X = group.drop(['thunder', 'datetime', 'city_name'], axis=1)
    y = group['thunder']

    X = X.to_numpy()
    y = y.to_numpy()

    dataset_ts[key] = (X, y)

class Objective:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __call__(self, trial):
        
        epochs = trial.suggest_int('epochs', 80, 700)
        hidden_size = trial.suggest_int('hidden_size', 25, 50)
        batch_size = trial.suggest_int('batch_size', 100, 512)
        model_name = trial.suggest_categorical('model_name', ['lstm', 'cnn_lstm'])
        scale_method = trial.suggest_categorical('scale_method', ['standard', 'minmax'])
        num_layers = trial.suggest_int('num_layers', 1, 5)
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        kernel_size = trial.suggest_int('kernel_size', 2, 4)
        num_cnn = trial.suggest_int('num_cnn', 1, 6)
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        n_steps = trial.suggest_int('n_steps', 20, 40)
            
        #with timesteps
        self.X = scale(self.X, scale_method)
        self.X, self.y = ts_preprocess(self.X, self.y, n_steps)
        self.X, self.y = balance(self.X, self.y)

        result_dict = lstm_torch_cv(self.X, self.y, epochs = epochs, hidden_size = hidden_size, batch_size=batch_size, model_name = model_name, num_layers=num_layers, dropout = dropout, kernel_size = kernel_size, num_cnn = num_cnn, lr =lr)

        return result_dict['f1_scores mean']
    

for point, dataset in dataset_ts.items():
        
    study = optuna.create_study(direction='maximize')
    study.optimize(Objective(*dataset), n_trials=100)

    best_params = study.best_params
    best_score = study.best_value

    best_params_with_score = {
        "best_params": best_params,
        "best_score": best_score
    }

    with open(f"best_params/{point}.json", "w") as f:
        json.dump(best_params_with_score, f, indent=4)

    print(f"Mejores parámetros guardados en 'best_params/{point}.json': {best_params_with_score}")
