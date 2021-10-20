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
        if frequency_data_columns is None:
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

    # def _split_single_experiment(self, experiment, targets, dataset):
    #     target = targets[targets['bearing_id'] == experiment]['status'].to_numpy()
    #     experiment_data = dataset[dataset['experiment_id'] == experiment]
    #     batch_time_range = (self.stable_area[0][1] - self.stable_area[0][0]) / self.splits_number
    #     experiment_prepared_vectors = None
    #     for split in range(self.splits_number):
    #
    #         batch = experiment_data[self.stable_area[0][0] + split * batch_time_range < experiment_data['timestamp']]
    #         batch = batch[batch['timestamp'] < self.stable_area[0][0] + (split + 1) * batch_time_range]
    #
    #         cleaned_vector = np.zeros(shape=(1, 1))
    #         cleaned_vector[0, 0] = target
    #
    #         for frequency_column in self.frequency_data_columns:
    #             frequency_data = batch[frequency_column]
    #             prepared_data = self._get_data_statistics(frequency_data.to_numpy())
    #             cleaned_vector = np.hstack((cleaned_vector, prepared_data))
    #
    #         if experiment_prepared_vectors is None:
    #             experiment_prepared_vectors = cleaned_vector
    #
    #         else:
    #             experiment_prepared_vectors = np.vstack((experiment_prepared_vectors, cleaned_vector))
    #
    #     return experiment_prepared_vectors
