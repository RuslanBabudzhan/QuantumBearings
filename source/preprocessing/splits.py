from typing import Tuple, List

import numpy as np
import pandas as pd
from scipy import fft

from source.datamodels.datamodels import Stats
from source.preprocessing.baseparser import BaseParser


class Splitter(BaseParser):
    def split_dataset(self,
                      dataset: pd.DataFrame,
                      targets: pd.DataFrame,
                      stable_area: Tuple[int, int] = (10, 20),
                      splits_number: int = 10,
                      frequency_data_columns: List[str] = None) -> np.ndarray:
        """
        Split dataset by chunks and return dataset with statistics of the chunks
        """
        self.stable_area = stable_area
        self.splits_number = splits_number
        self.frequency_data_columns = frequency_data_columns
        if self.frequency_data_columns is None:
            self.frequency_data_columns = ['a1_x', 'a1_y', 'a1_z', 'a2_x', 'a2_y', 'a2_z']
        experiments_indices = dataset['experiment_id'].unique()

        prepared_dataset = None

        for experiment in experiments_indices:

            experiment_prepared_vectors = self._split_single_experiment(experiment, targets, dataset)
            if prepared_dataset is None:
                prepared_dataset = experiment_prepared_vectors
            else:
                prepared_dataset = np.vstack((prepared_dataset, experiment_prepared_vectors))

        return prepared_dataset

    def _split_single_experiment(self, experiment, targets, dataset):
        target = targets[targets['bearing_id'] == experiment]['status'].to_numpy()
        experiment_data = dataset[dataset['experiment_id'] == experiment]

        batch_time_range = (self.stable_area[1] - self.stable_area[0]) / self.splits_number
        experiment_prepared_vectors = None
        for split in range(self.splits_number):

            batch = experiment_data[self.stable_area[0] + split * batch_time_range < experiment_data['timestamp']]
            batch = batch[batch['timestamp'] < self.stable_area[0] + (split + 1) * batch_time_range]

            cleaned_vector = np.zeros(shape=(1, 1))
            cleaned_vector[0, 0] = target

            for frequency_column in self.frequency_data_columns:
                frequency_data = batch[frequency_column]
                prepared_data = self._get_data_statistics(frequency_data.to_numpy())
                cleaned_vector = np.hstack((cleaned_vector, prepared_data))

            if experiment_prepared_vectors is None:
                experiment_prepared_vectors = cleaned_vector

            else:
                experiment_prepared_vectors = np.vstack((experiment_prepared_vectors, cleaned_vector))

        return experiment_prepared_vectors
