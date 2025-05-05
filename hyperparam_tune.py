import pandas as pd
import numpy as np
from scripts.preprocess import ts_preprocess, balance, scale
import optuna
import json
from scripts.train import lstm_torch_cv

dataframe = pd.concat([
    pd.read_csv("../data/climate_data_54.csv"),
    pd.read_csv("../data/climate_data_55.csv")
], ignore_index=True)

dataframe["datetime"] = pd.to_datetime(dataframe["datetime"])
dataframe_fs = dataframe.drop(['thunder_count', 'temp_min', 'temp_max', 'feels_like'], axis=1)

scale_cols = list(dataframe_fs.columns)
scale_cols = [x for x in scale_cols if x not in ['thunder', 'city_name', 'datetime']]

dataframe_fs[scale_cols] = scale(dataframe_fs[scale_cols], 'standard')
df_sorted = dataframe_fs.sort_values(by=["city_name", "datetime"])
groups = df_sorted.groupby('city_name')

def objective(trial):
    #hyperparameters
    
    epochs = trial.suggest_int('epochs', 80, 700)
    hidden_size = trial.suggest_int('hidden_size', 25, 50)
    batch_size = trial.suggest_int('batch_size', 100, 512)
    model_name = trial.suggest_categorical('model_name', ['lstm', 'cnn_lstm'])
    num_layers = trial.suggest_int('num_layers', 1, 5)
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
    kernel_size = trial.suggest_int('kernel_size', 2, 4)
    num_cnn = trial.suggest_int('num_cnn', 1, 6)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    n_steps = trial.suggest_int('n_steps', 20, 40)
    
    X_ts = []
    y_ts = []

    for key, group in groups:

        X = group.drop(['thunder', 'datetime', 'city_name'], axis=1)
        y = group['thunder']
        
        X = X.to_numpy()
        y = y.to_numpy()

        #with timesteps
        X, y = ts_preprocess(X, y, n_steps)
        X, y = balance(X, y)
        X_ts.append(X)
        y_ts.append(y)

    X_ts = np.concatenate(X_ts, axis=0)
    y_ts = np.concatenate(y_ts, axis=0)


    result_dict = lstm_torch_cv(X_ts, y_ts, epochs = epochs, hidden_size = hidden_size, batch_size=batch_size, model_name = model_name, num_layers=num_layers, dropout = dropout, kernel_size = kernel_size, num_cnn = num_cnn, lr =lr)

    return result_dict['accuracy means']

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

best_params = study.best_params
best_score = study.best_value

best_params_with_score = {
    "best_params": best_params,
    "best_score": best_score
}

with open("best_params.json", "w") as f:
    json.dump(best_params_with_score, f, indent=4)

print(f"Mejores parámetros guardados en 'best_params.json': {best_params_with_score}")
