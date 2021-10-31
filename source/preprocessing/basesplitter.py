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
        :param stats: list of statistics names, that will be calculated for each chunk for each dataset column
        :param use_signal: Either to use data of raw signals to generate features list or not
        :param use_specter: Either to use fft of raw signal to generate features list or not
        :param scaler: scaler with .fit_transform() method or None. Used for signal scaling
        :param specter_threshold: maximum frequency in Hz to calculate statistics if use_specter = True
        """
        self.use_signal = use_signal
        self.use_specter = use_specter

        self.scaler = scaler
        self.stable_area = None
        self.splits_number = None
        self.signal_data_columns = None
        self.specter_threshold = specter_threshold
        self.delta_time = None

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
        pass

    def _get_data_statistics(self, raw_data: np.ndarray):
        data = raw_data

        if self.scaler:
            data = self.scaler().fit_transform(X=data.T)
            data = data.T

        if self.use_signal and not self.use_specter:
            data = [data]
        elif self.use_specter and not self.use_signal:
            n = data.shape[1]
            fhat = fft.fft(data, n)
            PSD = fhat * np.conj(fhat)/n
            specter_threshold_index = int(self.specter_threshold*(self.delta_time*n))

            data = [PSD[:, :specter_threshold_index]]
        else:
            n = data.shape[1]
            fhat = fft.fft(data, n)
            PSD = fhat * np.conj(fhat)/n
            specter_threshold_index = int(self.specter_threshold*(self.delta_time*n))
            data = [data, PSD[:, :specter_threshold_index]]

        statistics_matrix = []
        for data_element in data:
            for statistic_function in self.stats.values():
                statistics_vector = statistic_function.count_stat(data_element, axis=1)
                statistics_matrix.append(statistics_vector)

        prepared = np.array(statistics_matrix).T
        return prepared
