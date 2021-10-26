from os import listdir
from os.path import isfile, join
import scipy.io
import pandas as pd
import numpy as np
from typing import List, Optional
from tqdm import tqdm

def convert_mat(path: str,
                column_names: List[str],
                thresh: int,
                label_1: str,
                add_keys_to_remove: Optional[List[str]] = []) -> pd.DataFrame:

    file_names = [file for file in listdir(path) if isfile(join(path, file))]
    keys_to_remove = ['__header__', '__version__', '__globals__']
    if add_keys_to_remove:
        keys_to_remove.append(*add_keys_to_remove)

    data = pd.DataFrame()

    for record in tqdm(file_names):
        rec_df = pd.DataFrame()
        exp = scipy.io.loadmat(path + record)
        for key in list(set(keys_to_remove) & set(exp.keys())):
            del exp[key]

        for col in exp.keys():
            rec_df = pd.concat([rec_df, pd.DataFrame(exp[col][:thresh])], axis=1)
        
        rec_df = pd.concat([rec_df, pd.DataFrame(record.replace('.mat', '').split('_')).T], axis=1).fillna(method='ffill')
        data = pd.concat([data, rec_df], axis=0)

    data.columns = column_names
    data['label'] = np.where(data['label'] == label_1 , 1, 0)
    data = data[data.columns.drop(list(data.filter(regex='del')))]

    return data


path = 'D:\Labs\Quantum\Bearings_luigi\VariableSpeedAndLoad\\' # path to .mat files

# columns in dataframe, depends on amount of bearings, axis and data in .mat files names
column_names = ['a1_x', 'a1_y', 'a1_z', 'a2_x', 'a2_y', 'a2_z', 'label', 'speed', 'load', 'del'] 
thresh = 12 # size of each record
label_without_defect = 'C0A' # how marked bearings without defect in file name, change to 1
luigi = convert_mat(path, column_names, thresh, label_1=label_without_defect)


path = 'D:\Labs\Quantum\Bearings_cesar\\'
column_names = ['a1_y', 'a2_y', 'a3_y', 'experiment_id', 'speed', 'del_1', 'del_2', 'label', 'del_3']
label_without_defect = 'F0'
cesar = convert_mat(path, column_names, thresh, label_1=label_without_defect, add_keys_to_remove=['Fs']) # Fs is not necessary, remove it


path = 'D:\Labs\Quantum\Bearings_cesar_2\\'
column_names = ['a1_y', 'a2_y', 'a3_y', 'experiment_id', 'speed', 'del_1', 'del_2', 'label', 'del_3']
cesar_2 = convert_mat(path, column_names, thresh, label_1=label_without_defect, add_keys_to_remove=['Fs'])
