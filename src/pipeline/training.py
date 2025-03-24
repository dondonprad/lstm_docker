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
from src.components.model import LSTM

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


@dataclass
class TrainingConfig():
     model_obj_path:str = os.path.join('artifacts', 'model.pkl')
     data_scaler = DataProcessingConfig().data_scaler

class Train():
     def __init__(self):
          self.training_config = TrainingConfig()
     
     def train(self, dataloader, model, loss_fn, optimizer):
          logging.info('training the dataset')
          try:
            loss_val = []
            model.train()
            for X_batch, y_batch in dataloader:
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch) # loss function using MSE
                loss_val.append(loss) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            save_object(self.training_config.model_obj_path, model)
            return loss_val
         
          except Exception as e:
                raise CustomException(sys,e)
          
     def test(self, X_test, y_test, model)->list:
          logging.info('test the model')
          try:
               model.eval()
               with torch.no_grad():
                    y_pred = model(X_test)
                    y_pred_rescaled = self.training_config.data_scaler.inverse_transform(y_pred)
                    y_test_rescaled = self.training_config.data_scaler.inverse_transform(y_test.reshape(-1,1))

                    # Evaluate model
                    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled) * 100
                    rmse = np.sqrt(mse) * 100
                    mape = mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled) * 100
                    wmape = (np.sum(np.abs(y_test_rescaled - y_pred_rescaled)) / np.sum(y_test_rescaled)) * 100
                    print(f'Test error: \n MSE:{mse:>0.1f}%, RMSE:{rmse:>0.1f}%, MAPE:{mape:>0.1f}%, WMAPE:{wmape:>0.1f}%')
               return [mse, rmse, mape, wmape]
                        
          except Exception as e:
               raise CustomException(sys,e)
          
     
     def fit(self, X_train, X_test, y_train, y_test, epochs):
          logging.info('train and validation')
          try:
               model_fn = LSTM(input_dim=1, hidden_dim=64, layer_dim=1, output_dim=1)
               optim_fn = optim.Adam(model_fn.parameters())
               loss_fn = nn.MSELoss()
          
               train_dataloader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)
          
               loss_train = []
               loss_test = []
               result = {}

               for epoch in range(epochs):
    
               #train model
                    model_fn.train()
                    for X_batch, y_batch in train_dataloader:
                         y_pred = model_fn(X_batch)
                         loss = loss_fn(y_pred, y_batch)
                         optim_fn.zero_grad()
                         loss.backward()
                         optim_fn.step()

                    # Validation
                    if epoch % 100 != 0:
                         continue
                    model_fn.eval()
                    with torch.no_grad():
                         #train loss
                         y_train_pred = model_fn(X_train)
                         #y_pred_train_rescaled = self.training_config.data_scaler.inverse_transform(y_train_pred.reshape(-1, 1))
                         #y_train_rescaled = self.training_config.data_scaler.inverse_transform(y_train.reshape(-1, 1))
                         train_rmse = np.sqrt(loss_fn(y_train_pred, y_train))
                         loss_train.append(train_rmse)
                         #test loss
                         y_test_pred = model_fn(X_test)
                         #y_pred_test_rescaled = self.training_config.data_scaler.inverse_transform(y_test_pred.reshape(-1, 1))
                         #y_test_rescaled = self.training_config.data_scaler.inverse_transform(y_test.reshape(-1, 1))
                         test_rmse = np.sqrt(loss_fn(y_test_pred, y_test))
                         loss_test.append(test_rmse)
                    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
               
               result['train'] = loss_train
               result['test'] = loss_test
               save_object(self.training_config.model_obj_path, model_fn)
               return result

          except Exception as e:
            raise CustomException(sys, e)




         


