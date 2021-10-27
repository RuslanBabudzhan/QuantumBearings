from os import listdir
from os.path import isfile, join
from threading import excepthook
import scipy.io
import pandas as pd
import numpy as np
from typing import List, Optional
from tqdm import tqdm

class Converter:

    @classmethod
    def path_to_list(cls, path):
        file_names = (file for file in listdir(path) if isfile(join(path, file)))
        for name in file_names:
            yield name


    @classmethod
    def cesar_convert(cls, 
                      path: str,
                      size: int) -> pd.DataFrame:

        keys_to_remove = ['__header__', '__version__', '__globals__', 'Fs', 'Rod_2']

        data = pd.DataFrame()
        file_names = cls.path_to_list(path)
        for record in tqdm(file_names):
            rec_df = pd.DataFrame(columns=['target', 'a1_y', 'a2_y', 'rpm', 'experiment_id'])
            exp = scipy.io.loadmat(path + record)
            for key in keys_to_remove:
                del exp[key]

            for key, col in zip(exp.keys(), ['a1_y', 'a2_y', 'a3_y']):
                rec_df[col] = pd.DataFrame(exp[key][:size])
            
            rec_df['target'], rec_df['rpm'], rec_df['experiment_id'] = record.split('_')[4], record.split('_')[1], int(record.split('_')[0])
            data = pd.concat([data, rec_df], axis=0)

        data['target'] = np.where(data['target'] == 'F0' , 1, 0)

        return data.reset_index(inplace=False)


    @classmethod
    def luigi_convert(cls,
                      path: str,
                      size: int) -> pd.DataFrame:
        
        keys_to_remove = ['__header__', '__version__', '__globals__']

        data = pd.DataFrame()
        file_names = cls.path_to_list(path)
        for i, record in tqdm(enumerate(file_names)):
            rec_df = pd.DataFrame(columns=['target', 'a1_x', 'a1_y', 'a1_z', 'a2_x', 'a2_y', 'a2_z', 'rpm', 'experiment_id'])
            exp = scipy.io.loadmat(path + record)

            for key in keys_to_remove:
                del exp[key]
            
            cols = ['a1_x', 'a1_y', 'a1_z', 'a2_x', 'a2_y', 'a2_z']
            rec_df[cols] = pd.DataFrame(exp[record.replace('.mat', '')][:size])
            rec_df['target'], rec_df['rpm'], rec_df['experiment_id'] = record.split('_')[0], record.split('_')[1], i
            data = pd.concat([data, rec_df], axis=0)

        data['target'] = np.where(data['target'] == 'F0' , 1, 0)

        return data.reset_index(inplace=False)

