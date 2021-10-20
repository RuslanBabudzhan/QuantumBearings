from typing import Tuple, List
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy import fft

from source.datamodels.datamodels import Stats


class BaseParser(ABC):
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

    @abstractmethod
    def split_dataset(self,
                      dataset: pd.DataFrame,
                      targets: pd.DataFrame,
                      stable_area: List[Tuple[int, int]],
                      splits_number: int,
                      frequency_data_columns: List[str] = None) -> np.ndarray:
        """
        Split dataset by chunks and return dataset with statistics of the chunks
        """
        pass

    @abstractmethod
    def _split_single_experiment(self, experiment, targets, dataset):
        """
        split by chunks one experiment records from whole experiments dataset
        :param experiment: experiment ID
        :param targets: targets dataset
        :param dataset: whole dataset to extract experiment data
        :return: statistics array with shape (splits, statistics) for of one splited experiment
        """
        pass

    def _get_data_statistics(self, raw_data):
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