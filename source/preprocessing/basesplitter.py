from typing import Tuple, List, Optional
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy import fft
from sklearn.preprocessing import StandardScaler

from source.datamodels.datamodels import Stats


class BaseSplitter(ABC):
    def __init__(self,
                 use_signal: bool = True,
                 use_specter: bool = False,
                 use_5_stats: bool = True,
                 use_15_stats: bool = False,
                 use_z_stat: bool = False):
        """
        Class implements chunk splittings of bearings_signals.csv dataset with subsequent processing of chunks
        """
        self.use_signal = use_signal
        self.use_specter = use_specter
        self.use_5_stats = use_5_stats
        self.use_15_stats = use_15_stats
        self.use_z_stat = use_z_stat
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

    @abstractmethod
    def split_dataset(self,
                      dataset: pd.DataFrame,
                      targets: pd.DataFrame,
                      stable_area: Optional[List[Tuple[int, int]]] = None,
                      splits_number: int = 10,
                      frequency_data_columns: List[str] = None) -> np.ndarray:
        """
        Split dataset by chunks and return dataset with statistics of the chunks
        """
        pass

    def _split_single_experiment(self, experiment, targets, dataset):
        """
        split by chunks one experiment records from whole experiments dataset
        :param experiment: experiment ID
        :param targets: targets dataset
        :param dataset: whole dataset to extract experiment data
        :return: statistics array with shape (splits, statistics) for of one splited experiment
        """
        target = targets[targets['bearing_id'] == experiment]['status'].values[0]
        experiment_data = dataset[dataset['experiment_id'] == experiment]
        batch_time_range = (self.stable_area[0][1] - self.stable_area[0][0]) / self.splits_number
        experiment_prepared_vectors = []
        for split in range(self.splits_number):

            batch = experiment_data[self.stable_area[0][0] + split * batch_time_range < experiment_data['timestamp']]
            batch = batch[batch['timestamp'] < self.stable_area[0][0] + (split + 1) * batch_time_range]

            cleaned_vector = [target]

            for frequency_column in self.frequency_data_columns:
                frequency_data = batch[frequency_column]
                prepared_data = self._get_data_statistics(frequency_data.to_numpy())
                cleaned_vector.extend(*prepared_data)
            experiment_prepared_vectors.append(cleaned_vector)
        return experiment_prepared_vectors

    def _get_data_statistics(self, raw_data):
        prepared = [0]
        data = raw_data

        assert self.use_signal or self.use_specter, "either use_signal or use_specter must be true"
        assert self.use_5_stats or self.use_15_stats, "either use_5_stats or use_15_stats must be true"

        if self.use_z_stat:
            data = (data - np.mean(data))/np.std(data)

        if self.use_signal and not self.use_specter:
            data = [data]
        elif self.use_specter and not self.use_signal:
            data = [np.abs(fft.fft(data))]
        else:
            data = [data, np.abs(fft.fft(data))]

        for data_element in data:
            statistic_values = []
            for statistic_function in self.stats.values():
                single_statistics_value = statistic_function.count_stat(data_element)
                print(type(single_statistics_value))
                statistic_values.append(single_statistics_value)

            if len(prepared) == 1:
                prepared[0] = statistic_values
            else:
                prepared = prepared.append(statistic_values)
        return prepared
