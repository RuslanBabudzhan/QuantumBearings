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
                      stable_area: Optional[List[Tuple[int, int]]] = None,
                      splits_number: int = 10,
                      signal_data_columns: List[str] = None) -> np.ndarray:
        """
        Split dataset by chunks and return dataset with statistics of the chunks
        """
        pass

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
