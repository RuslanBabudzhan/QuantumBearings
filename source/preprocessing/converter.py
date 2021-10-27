from os import listdir
from os.path import isfile, join

import scipy.io
import pandas as pd
import numpy as np
from tqdm import tqdm


class Converter:

    @classmethod
    def path_to_list(cls, path):
        file_names = (file for file in listdir(path) if isfile(join(path, file)))
        for name in file_names:
            yield name

    @classmethod
    def cesar_convert(cls, path: str) -> pd.DataFrame:

        keys_to_remove = ['__header__', '__version__', '__globals__', 'Fs', 'Rod_2']
        sample_rate = 40000

        data = pd.DataFrame()
        file_names = cls.path_to_list(path)
        for record in tqdm(file_names, total=45):
            rec_df = pd.DataFrame(columns=['target', 'a1_y', 'a2_y', 'rpm', 'experiment_id', 'timestamp'])
            exp = scipy.io.loadmat(path + record)
            for key in keys_to_remove:
                del exp[key]

            for key, col in zip(exp.keys(), ['a1_y', 'a2_y']):
                rec_df[col] = pd.DataFrame(exp[key])

            rec_name = record.split('_')
            rec_df['target'], rec_df['rpm'], rec_df['experiment_id'] = rec_name[4], rec_name[1], int(rec_name[0])
            time = int(len(rec_df) / sample_rate)
            rec_df['timestamp'] = np.linspace(0, time, sample_rate * time)
            data = pd.concat([data, rec_df], axis=0)

        data['target'] = np.where(data['target'] == 'F0', 1, 0)

        return data.reset_index(drop=True)

    @classmethod
    def luigi_convert(cls, path: str) -> pd.DataFrame:

        keys_to_remove = ['__header__', '__version__', '__globals__']
        sample_rate = 51200
        time = 10

        data = pd.DataFrame()
        file_names = cls.path_to_list(path)
        cols = ['a1_x', 'a1_y', 'a1_z', 'a2_x', 'a2_y', 'a2_z']
        for i in tqdm(range(119)):
            record = next(file_names)
            rec_df = pd.DataFrame(columns=['target', *cols, 'rpm', 'experiment_id', 'timestamp'])
            exp = scipy.io.loadmat(path + record)

            for key in keys_to_remove:
                del exp[key]

            rec_df[cols] = pd.DataFrame(exp[record.replace('.mat', '')])
            rec_df['target'], rec_df['rpm'], rec_df['experiment_id'] = record.split('_')[0], record.split('_')[1], i
            rec_df['timestamp'] = np.linspace(0, time, sample_rate * time)
            data = pd.concat([data, rec_df], axis=0)

        data['target'] = np.where(data['target'] == 'C0A', 1, 0)

        return data.reset_index(drop=True)
