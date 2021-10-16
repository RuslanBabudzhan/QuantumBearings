from typing import Tuple, List

import numpy as np
import pandas as pd
from scipy import fft

from source.datamodels.datamodels import Stats


class Splitter:
    def __init__(self,
                 use_signal: bool = True,
                 use_specter: bool = False,
                 use_5_stats: bool = True,
                 use_15_stats: bool = False):
        """
        Class implements chunk splittings of bearings_signals.csv dataset with subsequent processing of chunks
        """
        self.use_signal = use_signal
        self.use_specter = use_specter
        self.use_5_stats = use_5_stats
        self.use_15_stats = use_15_stats
        self.stable_area = None
        self.splits_number = None
        self.frequency_data_columns = None

        full_stats_list = Stats.get_keys()
        self.stats = set()
        if self.use_5_stats:
            for statistic_index in range(5):
                self.stats.add((
                    Stats[full_stats_list[statistic_index]].name,
                    Stats[full_stats_list[statistic_index]].value
                ))

        if self.use_15_stats:
            for statistic_index in range(2, 17):
                self.stats.add((
                    Stats[full_stats_list[statistic_index]].name,
                    Stats[full_stats_list[statistic_index]].value
                ))

        self.stats = dict(list(self.stats))

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

            experiment_prepared_vectors = self.__split_single_experiment(experiment, targets, dataset)
            if prepared_dataset is None:
                prepared_dataset = experiment_prepared_vectors
            else:
                prepared_dataset = np.vstack((prepared_dataset, experiment_prepared_vectors))

        return prepared_dataset

    def __split_single_experiment(self, experiment, targets, dataset):
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
                prepared_data = self.__prepare_data(frequency_data.to_numpy())
                cleaned_vector = np.hstack((cleaned_vector, prepared_data))

            if experiment_prepared_vectors is None:
                experiment_prepared_vectors = cleaned_vector

            else:
                experiment_prepared_vectors = np.vstack((experiment_prepared_vectors, cleaned_vector))

        return experiment_prepared_vectors

    def __prepare_data(self, raw_data):
        prepared = None

        assert self.use_signal or self.use_specter, "either use_signal or use_specter must be true"
        assert self.use_5_stats or self.use_15_stats, "either use_5_stats or use_15_stats must be true"

        if self.use_signal and not self.use_specter:
            data = [raw_data]
        elif self.use_specter and not self.use_signal:
            data = [np.abs(fft.fft(raw_data))]
        else:
            data = [raw_data, np.abs(fft.fft(raw_data))]

        for data_element in data:
            statistic_values = []
            for statistic_function in self.stats.values():
                statistic_values.append(statistic_function.count_stat(data_element))
            statistic_values = np.array(statistic_values)

            if prepared is None:
                prepared = statistic_values.reshape(1, -1)
            else:
                prepared = np.hstack((prepared, statistic_values.reshape(1, -1)))
        return prepared