import pandas as pd
import numpy as np
from scripts.preprocess import ts_preprocess, balance, scale
import optuna
import json
from scripts.train import lstm_torch_cv
import os
from sklearn.model_selection import train_test_split
from scripts.train import lstm_torch_train
import torch
import os

os.makedirs("trained_models", exist_ok=True)

data_dirs = os.listdir("data")
dfs = [pd.read_csv(f"data/{dir}") for dir in data_dirs]
dataframe = pd.concat(dfs, ignore_index=True)

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



for point, dataset in dataset_ts.items():

    #load the best hyper params 
    with open(f'scripts/best_params/{point}.json') as f:
        best = json.load(f)
    best = best['best_params']

    scale_method = best.pop('scale_method')
    n_steps = best.pop('n_steps')

    X, y = dataset

    X = scale(X, scale_method)
    X, y = ts_preprocess(X, y, n_steps)
    X, y = balance(X, y)

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    result, model = lstm_torch_train(**best, X_train=X_train, X_val=X_test, y_train=y_train, y_val=y_test, return_model=True)

    torch.save(model.state_dict(), f'trained_models/{point}.pth')
    with open(f'trained_models/{point}_result.txt', 'w') as f:
        f.write(str(result))

    print(result)


