from torch.utils.data import TensorDataset, DataLoader
import lightning as L
import torch 
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
from scripts.model import LightningLSTM, CNN_LSTM


def test_model(X_test, y_test, model):
    with torch.no_grad():

        y_pred = model(torch.tensor(X_test, dtype=torch.float32)).cpu().numpy()
        y_pred = y_pred.reshape(y_pred.shape[0])
        y_pred_ = y_pred.copy()
        # fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        # optimal_idx = np.argmax(tpr - fpr)
        # optimal_threshold = thresholds[optimal_idx]
        

        # precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred)
        # differences = np.abs(precisions - recalls)
        # optimal_threshold = thresholds[np.argmin(differences)]
        
        # thresholds = np.linspace(y_pred.min(), y_pred.max(), 100)
        # f1_scores = [f1_score(y_test, y_pred >= t) for t in thresholds]
        # optimal_threshold = thresholds[np.argmax(f1_scores)]

        optimal_threshold = 0.5
        y_pred[ y_pred > optimal_threshold] = 1
        y_pred[ y_pred <= optimal_threshold] = 0
        
#        y_pred = torch.round(y_pred).numpy()
        #y_pred = y_pred.reshape(y_pred.shape[0])

        # print(y_test)
        # print(y_pred)
        # print(y_pred_)
        return accuracy_score(y_test, y_pred), \
                f1_score(y_test, y_pred), \
                precision_score(y_test, y_pred), \
                recall_score(y_test, y_pred), \
                roc_auc_score(y_test, y_pred), \
                optimal_threshold
    
def lstm_torch_train(X_train, X_val, y_train, y_val, epochs = 700, hidden_size = 32, batch_size=128, model_name = 'lstm', num_layers=2, dropout = 0.2, kernel_size = 3, num_cnn = 4, lr = 1e-3):

        inputs = torch.tensor(X_train, dtype=torch.float32)
        labels = torch.tensor(y_train, dtype=torch.float32)
        labels = labels.reshape(labels.shape[0], 1)

        dataset = TensorDataset(inputs, labels) 
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        if model_name == 'lstm':
            model = LightningLSTM(X_train.shape[2], hidden_size=hidden_size, num_layers=num_layers, dropout = dropout, lr = lr)
        elif model_name == 'cnn_lstm':
            model = CNN_LSTM(X_train.shape[2], hidden_size=hidden_size, num_layers=num_layers, dropout = dropout, kernel_size = kernel_size, num_cnn = num_cnn, lr = lr)
        else:
            raise Exception
        trainer = L.Trainer(max_epochs=epochs, log_every_n_steps=2)

        trainer.fit(model, train_dataloaders=dataloader)

        return test_model(X_val, y_val, model)
        


def lstm_torch_cv(X, y, epochs = 700, hidden_size = 32, batch_size=128, random_state=42, cv = 4, model_name = 'lstm', num_layers=2, dropout = 0.2, kernel_size = 3, num_cnn = 4, lr = 1e-3):

    kfolds = StratifiedKFold(n_splits=cv, random_state=random_state, shuffle=True)

    result_dict = {
        "accuracies": [],
        "f1_scores": [],
        "precisions" : [],
        "recalls" : [],
        "aucs": [],
        "optimal_thresholds" : []
    }
    
    for train_index, test_index in kfolds.split(X, y):

        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]

        accuracy, f1, precision, recall, auc, optimal_threshold = lstm_torch_train(
                X_train, X_val, y_train, y_val, epochs = epochs, hidden_size = hidden_size, batch_size=batch_size, model_name = model_name, num_layers=num_layers, dropout = dropout, kernel_size = kernel_size, num_cnn = num_cnn, lr = lr
            )
        
        result_dict["accuracies"].append(accuracy)
        result_dict["f1_scores"].append(f1)
        result_dict["precisions"].append(precision)
        result_dict["recalls"].append(recall)
        result_dict["aucs"].append(auc)
        result_dict["optimal_thresholds"].append(optimal_threshold)
    result_dict["accuracy mean"] = np.mean(result_dict["accuracies"])
    result_dict["f1_scores mean"] = np.mean(result_dict["f1_scores"])
    result_dict["precisions mean"] = np.mean(result_dict["precisions"])
    result_dict["recalls mean"] = np.mean(result_dict["recalls"])
    result_dict["aucs mean"] = np.mean(result_dict["aucs"])
    result_dict["optimal_thresholds mean"] = np.mean(result_dict["optimal_thresholds"])
    for k, v in result_dict.items():
        print(k, v)

    return result_dict