import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from src.components.data_preprocessing import DataProcessingConfig
from src.components.model import LSTM, LSTM2

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


@dataclass
class TrainingConfig():
     model_obj_path:str = os.path.join('artifacts', 'model.pkl')
     result_path:str = os.path.join('artifacts','result.pkl')
     data_scaler = DataProcessingConfig().data_scaler

class Train():
     def __init__(self):
          self.training_config = TrainingConfig()
     
     def fit(self, X_train, X_test, y_train, y_test, epochs):
          logging.info('train and validation')
          try:
               model_fn = LSTM(input_dim=1, hidden_dim=64, layer_dim=1, output_dim=1)
               optim_fn = optim.Adam(model_fn.parameters())
               loss_fn = nn.MSELoss()
          
               dataloader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)
          
               loss_train = []
               loss_test = []
               result = {}

               for epoch in range(epochs):
    
               #train model
                    model_fn.train()
                    for X_batch, y_batch in dataloader:
                         y_pred = model_fn(X_batch)
                         loss = loss_fn(y_pred, y_batch)
                         optim_fn.zero_grad()
                         loss.backward()
                         optim_fn.step()

                    # Validation
                    if epoch %100 != 0:
                         continue #print(f'Epoch [{epoch}/{epochs}], loss: {loss.item():.4f}')

                    model_fn.eval()
                    with torch.no_grad():
                         #train loss
                         y_train_pred = model_fn(X_train)
                         y_train_pred_rescaled = torch.tensor(self.training_config.data_scaler.inverse_transform(y_train_pred))
                         y_train_rescaled = torch.tensor(self.training_config.data_scaler.inverse_transform(y_train.reshape(-1, 1)))
                         train_rmse = np.sqrt(loss_fn(y_train_pred_rescaled, y_train_rescaled).detach().numpy())
                         loss_train.append(train_rmse)
                         #test loss
                         y_test_pred = model_fn(X_test)
                         y_test_pred_rescaled = torch.tensor(self.training_config.data_scaler.inverse_transform(y_test_pred))
                         y_test_rescaled = torch.tensor(self.training_config.data_scaler.inverse_transform(y_test.reshape(-1, 1)))
                         test_rmse = np.sqrt(loss_fn(y_test_pred_rescaled, y_test_rescaled).detach().numpy())
                         loss_test.append(test_rmse)

                    print(f'Epoch [{epoch}/{epochs}], loss: {loss.item():.4f}')
                    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
               
               result['train'] = loss_train
               result['test'] = loss_test
               save_object(self.training_config.model_obj_path, model_fn)
               #save_object(self.training_config.result_path,result)
               return result

          except Exception as e:
            raise CustomException(sys, e)
          
class Train2():
     def __init__(self):
          self.training_config = TrainingConfig()

     def fit(self, X_train, X_test, y_train, y_test, epochs):
          logging.info('train and validation')
          try:
               model_fn = LSTM2(input_dim=1, hidden_dim=50, layer_dim=1, output_dim=1)
               optim_fn = optim.Adam(model_fn.parameters(), lr=0.01)
               loss_fn = nn.MSELoss()
          
               val_train = []
               val_test = []
               result = {}

               for epoch in range(epochs):
                    output = model_fn(X_train.unsqueeze(-1)).squeeze()
                    optim_fn.zero_grad()
                    loss = loss_fn(output, y_train)
                    loss.backward()
                    optim_fn.step()

                    if (epoch + 1) %10 == 0:
                         print(f'Epoch [{epoch+1}/{epochs}], loss: {loss.item():.4f}')

                    model_fn.eval()
                    with torch.no_grad():
                         #train val
                         y_train_pred = model_fn(X_train)
                         y_train_pred_rescaled = self.training_config.data_scaler.inverse_transform(y_train_pred)
                         y_train_rescaled = self.training_config.data_scaler.inverse_transform(y_train)
                         train_rmse = np.sqrt(loss_fn(y_train_pred_rescaled, y_train_rescaled))
                         val_train.append(train_rmse)
                         #test loss
                         y_test_pred = model_fn(X_test)
                         y_test_pred_rescaled = self.training_config.data_scaler.inverse_transform(y_test_pred.reshape(-1, 1))
                         y_test_rescaled = self.training_config.data_scaler.inverse_transform(y_test.reshape(-1, 1))
                         test_rmse = np.sqrt(loss_fn(y_test_pred_rescaled, y_test_rescaled))
                         val_test.append(test_rmse)
                    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
               
               result['train'] = val_train
               result['test'] = val_test
               save_object(self.training_config.model_obj_path, model_fn)
               return result





          except Exception as e:
               raise CustomException(sys,e)






         


