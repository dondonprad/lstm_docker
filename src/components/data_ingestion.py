import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

@dataclass
class DataFrameConfig():
    dataset_path:str = os.path.join('artifacts', 'dataset.csv')

class DataFrame():
    def __init__ (self):
        self.data_set_config = DataFrameConfig()

    def dataset(self)->list:
        # Create DataFrame from provided dataset
        logging.info('create dataset')
        try: 
            data = {
                    "Date": pd.date_range(start="2022-06-19", periods=31, freq='D'),
                    "Pods": [3090, 3088, 3088, 3088, 3120, 2910, 1210, 3090, 3088, 3088, 3088, 3120, 
                             2910, 1210, 3090, 3088, 3086, 3086, 3120, 2910, 1210, 3100, 3106, 3106, 
                             3106, 3120, 2910, 1210, 3120, 3120, 3120]
                   }
            df = pd.DataFrame(data)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.to_csv(self.data_set_config.dataset_path, index=True, header=True)
            return df

        except Exception as e:
            raise CustomException(e,sys)

    
if __name__ == '__main__':
    df= DataFrame().dataset()
    print(df)