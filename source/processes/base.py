from enum import Enum
from typing import List, Union, Optional
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModeler(ABC):
    """
    Class implements bootstrap resampling of test dataset to build a scores distribution for specific models
    """

    RANDOM_STATE = 42
    __RANDOMIZER = np.random.RandomState(RANDOM_STATE)

    @abstractmethod
    def initialize_logging(self,
                           result_label_prefix: str,
                           logging_type: str = 'experiment',
                           log_folder: Optional[str] = None,
                           ):
        """initialize log properties"""
        pass

    @abstractmethod
    def run(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray],
            bearing_positive_ID: Optional[np.array] = None,
            bearing_negative_ID: Optional[np.array] = None,
            verbose: bool = False):
        """method to run experiment"""

    @staticmethod
    def _generate_groups_from_id(bearings_id, group_size):
        return np.array([np.arange(group * group_size, (group + 1) * group_size) for group in bearings_id]).flatten()

    @staticmethod
    def _split_test_train(X: np.ndarray, y: np.ndarray, train_ID: np.ndarray, test_ID: np.ndarray):
        group_size = int(X.shape[0] / (len(train_ID) + len(test_ID)))
        train_groups = BaseModeler._generate_groups_from_id(train_ID, group_size)
        test_groups = BaseModeler._generate_groups_from_id(test_ID, group_size)

        x_train = X[train_groups]
        y_train = y[train_groups]
        x_test = X[test_groups]
        y_test = y[test_groups]
        return x_train, x_test, y_train, y_test

    @staticmethod
    def _get_test_train_indices(id_positive: np.ndarray, id_negative: np.ndarray, train_ratio: float):
        permuted_positive = BaseModeler.__RANDOMIZER.permutation(id_positive)
        permuted_negative = BaseModeler.__RANDOMIZER.permutation(id_negative)

        train_positive_idx = permuted_positive[:int(len(id_positive) * train_ratio)]
        train_negative_idx = permuted_negative[:int(len(id_negative) * train_ratio)]

        test_positive_idx = permuted_positive[int(len(id_positive) * train_ratio):]
        test_negative_idx = permuted_negative[int(len(id_negative) * train_ratio):]

        train_idx = BaseModeler.__RANDOMIZER.permutation(np.concatenate((train_positive_idx, train_negative_idx)))
        test_idx = BaseModeler.__RANDOMIZER.permutation(np.concatenate((test_positive_idx, test_negative_idx)))
        return train_idx, test_idx

    @abstractmethod
    def _create_separate_log_file(self,  result_label, result, model_name, resample_id,
                                  train_indices, test_indices):
        """generates log file for one fit in experiment"""
        pass

    @abstractmethod
    def _create_experiment_log_file(self,  result_label, results, train_ID, test_ID):
        """generates log file for whole experiment"""
        pass


class Logging(Enum):
    """logging type"""
    silent = 1
    experiment = 2
    separated = 3
    all = 4

    @staticmethod
    def get_keys() -> List[str]:
        return list(map(lambda c: c.name, Logging))
