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
from src.utils import save_object
from dataclasses import dataclass

import warnings
warnings.filterwarnings('ignore')

@dataclass
class MainConfig():
    result_path:str = os.path.join('artifacts','results.pkl')

class Main():
    def __init__(self):
        self.data_frame = DataFrame()
        self.data_preprocessing = DataProcessing()
        self.training = Train()
        self.main_config = MainConfig()

    def main(self):
        #initiate dataset 
        try: 
            logging.info('data training initiate')
            data = self.data_frame.dataset()

            #data preprocessing
            data_set = data['Pods'].values.astype('float32')
            splits = [0.60, 0.70, 0.80]
            results = {}
        
            for split in splits:
                #train
                epoch = 8000
                print('===============================================')
                print(f'Training using:{split} data ratio')
                print('------------------------------------------------')
                X_train, X_test, y_train, y_test = self.data_preprocessing.initiate_create_split_scaled_dataset(data_set, split) 
                result = self.training.fit(X_train, X_test, y_train, y_test, epoch)
                results[str(f'{split*100}%')] = result
                print('===============================================')

            save_object(self.main_config.result_path,results)
            #return results

        except Exception as e:
            raise CustomException(sys, e)

if __name__ == '__main__':
    Main().main()
    #print(result)
        



