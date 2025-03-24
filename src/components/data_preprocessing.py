import os
import sys
import pandas as pd
import numpy as np
import torch
from dataclasses import dataclass

from sklearn.preprocessing import MinMaxScaler

from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataFrame
from src.utils import save_object

@dataclass
class DataProcessingConfig():
    data_processing_obj_path:str = os.path.join('artifacts', 'processor.pkl')
    data_scaler = MinMaxScaler()


class DataProcessing():
    def __init__(self):
        self.data_processing_config = DataProcessingConfig()

    def initiate_data_scaler(self, train, test): #scale data MinMaxscaler
        logging.info('Data Scaler Processing')
        try:
            # Normalize the data
            scaler = self.data_processing_config.data_scaler
            train_scaled = scaler.fit_transform(np.array(train).reshape(-1,1))
            test_scaled = scaler.transform(np.array(test).reshape(-1,1))
            # save object datascaler
            save_object(self.data_processing_config.data_processing_obj_path, scaler)
            return train_scaled, test_scaled

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_dataset(self, data, lookback): #create dataset create X, y
        logging.info('Create squence from data frame')
        try: 
            """
            Transform a time series into a prediction dataset
    
                Args:
                dataset: A numpy array of time series, first dimension is the time steps
                lookback: Size of window for prediction
            """
            X, y = [], []
            for i in range(len(data)-lookback):
                feature = data[i:i+lookback]
                target = data[i+1:i+lookback+1]
                X.append(feature)
                y.append(target)

            return torch.tensor(X), torch.tensor(y)
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_create_split_scaled_dataset(self, data, train_rate): #split data into train and test then scale using MinMax
        logging.info('Split train test dataset')
        try:
            lookback = 1
            train_size = int(len(data) * train_rate)
            #test_size = len(data) - train_size
            train, test = data[:train_size], data[train_size:]
            #train_scaled, test_scaled = self.initiate_data_scaler(train, test)
            X_train, y_train = self.initiate_dataset(train, lookback)
            X_test, y_test = self.initiate_dataset(test, lookback)

            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == '__main__':
    obj = DataProcessing()
    df = pd.read_csv('/home/server-iss-mbkm/project/docker/LSTM/artifacts/dataset.csv')
    data = df['Pods'].values.astype('float32')
    X_train, X_test, y_train, y_test = obj.initiate_create_split_scaled_dataset(data, 0.70)
    print(X_train.shape)
    print(X_test.shape)

