import numpy as np
import pandas as pd
from scipy import stats, fft

from scipy.stats import kurtosis, skew, variation, iqr
from pyentrp import entropy as pent
import antropy as ent
from hurst import compute_Hc

from typing import Tuple, List

from dataclasses import dataclass

from mljson import Stats


def crest_factor(x):
    return np.max(np.abs(x)) / np.sqrt(np.mean(np.square(x)))


def get_stat_features(data):
    H, c, _ = compute_Hc(data, kind='change')
    activity, complexity = ent.hjorth_params(data)
    return np.array([variation(data),
                     kurtosis(data),
                     skew(data),
                     (max(data) - min(data)),
                     iqr(data),
                     float(pent.sample_entropy(data, 1)),
                     pent.shannon_entropy(data),
                     sum(np.abs(data) ** 2),
                     H,
                     ent.petrosian_fd(data),
                     ent.num_zerocross(data),
                     ent.higuchi_fd(data),
                     activity,
                     complexity,
                     crest_factor(data)])


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
                print(statistic_index)
                self.stats.add(Stats[full_stats_list[statistic_index]])

        if self.use_15_stats:
            for statistic_index in range(2, 17):
                print(statistic_index)
                self.stats.add(Stats[full_stats_list[statistic_index]])


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

            experiment_prepared_vectors = self.__split_experiment(experiment, targets, dataset)
            if prepared_dataset is None:
                prepared_dataset = experiment_prepared_vectors
            else:
                prepared_dataset = np.vstack((prepared_dataset, experiment_prepared_vectors))

        return prepared_dataset

    def __split_experiment(self, experiment, targets, dataset):
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
            # TODO: rewrite for using self.stats
            if self.use_5_stats:
                statistics_5 = [np.mean(data_element),
                                np.std(data_element),
                                stats.kurtosis(data_element),
                                stats.skew(data_element),
                                stats.variation(data_element)]
                statistics_5 = np.array(statistics_5)

                if prepared is None:
                    prepared = statistics_5.reshape(1, -1)
                else:
                    prepared = np.hstack((prepared, statistics_5.reshape(1, -1)))

            if self.use_15_stats:
                statistics_15 = get_stat_features(data_element)
                if prepared is None:
                    prepared = statistics_15.reshape(1, -1)
                else:
                    prepared = np.hstack((prepared, statistics_15.reshape(1, -1)))
        return prepared


splitter = Splitter(use_specter=True, use_5_stats=True, use_15_stats=True)