from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from scipy import fft

from source.datamodels.datamodels import Stats
from source.preprocessing.basesplitter import BaseSplitter


class Splitter(BaseSplitter):
    DATA_REQUIRED_COLUMNS = ['experiment_id', 'timestamp']
    TARGET_REQUIRED_COLUMNS = ['bearing_id', 'status']

    def split_dataset(self,
                      dataset: pd.DataFrame,
                      targets: pd.DataFrame,
                      stable_area: Optional[List[Tuple[int, int]]] = None,
                      splits_number: int = 10,
                      frequency_data_columns: List[str] = None) -> np.ndarray:
        """
        Split dataset by chunks and return dataset with statistics of the chunks
        """
        self.splits_number = splits_number
        if stable_area is None:
            stable_area = [(10, 20)]
        self.stable_area = stable_area
        if frequency_data_columns is None:  # TODO: frequency data can`t be overwrite here if obj already has this data
            frequency_data_columns = ['a1_x', 'a1_y', 'a1_z', 'a2_x', 'a2_y', 'a2_z']
        self.frequency_data_columns = frequency_data_columns

        required_data_columns = Splitter.DATA_REQUIRED_COLUMNS.copy()
        required_data_columns.extend(frequency_data_columns)
        for column in required_data_columns:
            if column not in dataset.columns:
                raise ValueError(f"dataset must have {column} column")

        for column in Splitter.TARGET_REQUIRED_COLUMNS:
            if column not in targets.columns:
                raise ValueError(f"target must have {column} column")

        experiments_indices = dataset['experiment_id'].unique()

        prepared_dataset = None

        for experiment in experiments_indices:

            experiment_prepared_vectors = self._split_single_experiment(experiment, targets, dataset)
            if prepared_dataset is None:
                prepared_dataset = experiment_prepared_vectors
            else:
                prepared_dataset = np.vstack((prepared_dataset, experiment_prepared_vectors))

        return prepared_dataset

signals_dataset = pd.read_csv('F:/PythonNotebooks/Study/Quantum/Bearings/data/own datasets/bearing_signals.csv')
classes_dataset = pd.read_csv('F:/PythonNotebooks/Study/Quantum/Bearings/data/own datasets/bearing_classes.csv', delimiter=';', skiprows=[1])

splitter = Splitter(use_specter=True, use_5_stats=True, use_15_stats=False, use_z_stat=False)
prepared_data = splitter.split_dataset(signals_dataset, classes_dataset)

print(f"features number: {prepared_data.shape[1]-1}")
print(f"examples number: {prepared_data.shape[0]}")
prepared_dataframe = pd.DataFrame(prepared_data, columns=None)

X = prepared_dataframe.iloc[:, 1:]
y = prepared_dataframe.iloc[:, 0]

print(X.head())
