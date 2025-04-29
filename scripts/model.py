import torch.nn as nn
import torch.nn.functional as F 
from torch.optim import Adam

import lightning as L

class LightningLSTM(L.LightningModule):

    def __init__(self, input_size, hidden_size=32, output_size=1, num_layers=2, dropout = 0.2, lr = 1e-3): 
        
        super().__init__()

        L.seed_everything(seed=42)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout,
            num_layers=num_layers,
            ) 
        self.hidden2label = nn.Linear(hidden_size, output_size)
        self.criterion = nn.BCELoss()
        self.input_size = input_size
        self.lr = lr
         

    def forward(self, input):

        lstm_out, temp = self.lstm(input)

        prediction = lstm_out[:,-1, :] 
        prediction = self.hidden2label(prediction)
        prediction = F.sigmoid(prediction)
        return prediction
        
        
    def configure_optimizers(self): 
        return Adam(self.parameters(), lr=self.lr) 

    
    def training_step(self, batch, batch_idx): 
        input_i, label_i = batch 
        output_i = self(input_i)
        #loss = (output_i - label_i)**2 
        loss = self.criterion(output_i, label_i)

        self.log("train_loss", loss)

        return loss
    
class CNN_LSTM(L.LightningModule):
    def __init__(self, input_size, hidden_size=32, num_classes = 1, num_layers=2, dropout = 0.2, kernel_size = 3, num_cnn = 4, lr = 1e-3):
        super().__init__()

        cnn_layers = []
        in_channels = input_size
        out_channels = 32

        for i in range(num_cnn):
            cnn_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=1))
            cnn_layers.append(nn.ReLU())
            in_channels = out_channels
            out_channels *= 2      

        self.cnn = nn.Sequential(*cnn_layers)

        self.lstm = nn.LSTM(input_size=out_channels//2, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.criterion = nn.BCELoss()
        self.lr = lr

    def forward(self, x):
        #cnn takes input of shape (batch_size, channels, seq_len)
        x = x.permute(0, 2, 1)
        out = self.cnn(x)
        # lstm takes input of shape (batch_size, seq_len, input_size)
        out = out.permute(0, 2, 1)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        out = F.sigmoid(out)
        return out
            
    def configure_optimizers(self): 
        return Adam(self.parameters(), lr=self.lr) 

    
    def training_step(self, batch, batch_idx): 
        input_i, label_i = batch 
        output_i = self(input_i)
        #loss = (output_i - label_i)**2 
        loss = self.criterion(output_i, label_i)

        self.log("train_loss", loss)

        return loss