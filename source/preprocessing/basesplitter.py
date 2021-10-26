from typing import Tuple, List, Optional, Any
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy import fft

from source.datamodels.iterators import Stats


class BaseSplitter(ABC):
    def __init__(self,
                 stats: List[str],
                 use_signal: bool = True,
                 use_specter: bool = False,
                 scaler: Optional[Any] = None,
                 specter_threshold: Optional[int] = None):
        """
        Class implements chunk splittings of bearings_signals.csv dataset with subsequent processing of chunks
        """
        self.use_signal = use_signal
        self.use_specter = use_specter

        self.scaler = scaler
        self.stable_area = None
        self.splits_number = None
        self.signal_data_columns = None
        self.specter_threshold = specter_threshold

        full_stats_list = Stats.get_keys()
        if not set(stats).issubset(full_stats_list):
            raise ValueError(f"Invalid statistics. Possible values: {full_stats_list}")

        self.stats = set()
        for statistic_name in full_stats_list:
            if statistic_name in stats:
                self.stats.add((
                    statistic_name,
                    Stats[statistic_name].value
                ))
        self.stats = dict(list(self.stats))

    @abstractmethod
    def split_dataset(self,
                      dataset: pd.DataFrame,
                      targets: pd.DataFrame,
                      stable_area: Optional[List[Tuple[int, int]]] = None,
                      splits_number: int = 10,
                      signal_data_columns: List[str] = None) -> np.ndarray:
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

            for frequency_column in self.signal_data_columns:
                frequency_data = batch[frequency_column]
                prepared_data = self._get_data_statistics(frequency_data.to_numpy())
                cleaned_vector.extend(*prepared_data)
            experiment_prepared_vectors.append(cleaned_vector)
        return experiment_prepared_vectors

    def _get_data_statistics(self, raw_data: np.ndarray):
        data = raw_data

        if self.scaler:
            data = self.scaler().fit_transform(X=data.T)
            data = data.T

        if self.use_signal and not self.use_specter:
            data = [data]
        elif self.use_specter and not self.use_signal:
            data = [np.abs(fft.fft(data, axis=1))[:, :self.specter_threshold]]
        else:
            data = [data, np.abs(fft.fft(data, axis=0))[:, :self.specter_threshold]]

        statistics_matrix = []
        for data_element in data:
            for statistic_function in self.stats.values():
                statistics_vector = statistic_function.count_stat(data_element, axis=1)
                statistics_matrix.append(statistics_vector)

        prepared = np.array(statistics_matrix).T
        return prepared

