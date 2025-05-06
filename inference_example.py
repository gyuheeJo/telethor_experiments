import pandas as pd
import numpy as np
from scripts.preprocess import ts_preprocess, scale
import json
from scripts.model import CNN_LSTM, LightningLSTM
import torch

dataframe = pd.read_csv('data/climate_data_54.csv')

#load hyper params
with open(f'scripts/best_params/54_f1.json') as f:
    best = json.load(f)
best = best['best_params']

#convertimos a formato de datetime
dataframe["datetime"] = pd.to_datetime(dataframe["datetime"])

#quitamos las columnas redundantes
dataframe_fs = dataframe.drop(['thunder_count', 'temp_min', 'temp_max', 'feels_like'], axis=1)

#ordenamos por datetime
df_sorted = dataframe_fs.sort_values(by=["datetime"])

#eliminamos columnas innecesarias para la predicción y seperamaos X y Y
X = df_sorted.drop(['thunder', 'datetime', 'city_name'], axis=1).to_numpy()
y = df_sorted['thunder'].to_numpy()

scale_method = best.pop('scale_method')
n_steps = best.pop('n_steps')

#normalizamos los datos
X = scale(X, scale_method)
#convertimos en en shape (N, seq_length, n_columns)
X, y = ts_preprocess(X, y, n_steps)

#load model
if best['model_name'] == 'cnn_lstm':
    model = CNN_LSTM(X.shape[2], hidden_size=best['hidden_size'], num_layers=best['num_layers'], dropout=best['dropout'], kernel_size=best['kernel_size'], num_cnn=best['num_cnn'])
elif best['model_name'] == 'lstm':
    model = LightningLSTM(X.shape[2], hidden_size=best['hidden_size'], num_layers=best['num_layers'], dropout=best['dropout'])
else:
    raise Exception
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load('trained_models/54.pth',
                        map_location=torch.device(device),
                        weights_only=True)

model.load_state_dict(checkpoint)
model.eval()

#predecimos
y_pred = model(torch.tensor(X, dtype=torch.float32)).cpu().detach().numpy()
y_pred = y_pred.reshape(y_pred.shape[0])
y_pred_logit = y_pred.copy()

optimal_threshold = 0.5
y_pred[ y_pred > optimal_threshold] = 1
y_pred[ y_pred <= optimal_threshold] = 0

#mostramos 30 primeros
print(f"preditions logits:\n{y_pred_logit[:30]}")
print(f"preditions:\n{y_pred[:30].astype(np.int32)}")
print(f"reals:\n{y[:30]}")