import os 
import sys
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from src.components.data_preprocessing import DataProcessing
from src.components.data_ingestion import DataFrame
from src.pipeline.training import Train
from src.logger import logging
from src.exception import CustomException

class Main():
    def __init__(self):
        self.data_frame = DataFrame()
        self.data_preprocessing = DataProcessing()
        self.training = Train()

    def main(self):
        #initiate dataset 
        try: 
            logging.info('data training initiate')
            data = self.data_frame.dataset()

            #data preprocessing
            data_set = data['Pods'].values.astype('float32')
            X_train, X_test, y_train, y_test = self.data_preprocessing.initiate_create_split_scaled_dataset(data_set, 0.70) 

            #train
            epoch = 500
            result = self.training.fit(X_train, X_test, y_train, y_test, epoch)

            return result

        except Exception as e:
            raise CustomException(sys, e)

if __name__ == '__main__':
    result= Main().main()
    print(result)
        



