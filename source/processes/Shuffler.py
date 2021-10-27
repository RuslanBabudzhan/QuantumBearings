from typing import Union, Generator
import numpy as np
import pandas as pd


def id_shuffler(data: Union[pd.DataFrame, np.ndarray],
                train_size: float) -> np.ndarray:
    """
    Shuffle given array into train and test datasets

    data: array of indices
    train_size: size of train dataset
    """
    len_data = len(data)
    N = int(len_data * train_size)
    data = np.array(data)
    np.random.shuffle(data)
    train, test = data[:N], data[N:]

    return train, test


class OverlapGroupCV():

    def __init__(self,
                 train_size: float,
                 n_repeats: int = 100):

        assert 0 <= train_size <= 1, 'Train size should be between 0 and 1'
        self.train_size = train_size
        self.n_repeats = n_repeats

    def split(self,
              X: Union[pd.DataFrame, np.ndarray],
              y: Union[pd.DataFrame, np.ndarray],
              groups: Union[pd.DataFrame, np.ndarray]) -> Generator:
        """
        Create index generator for cross-validation

        groups: data['experiment_id']
        """
        assert len(y) == len(groups), (
            "Length of the predictors is not"
            "matching with the groups.")

        id_train_array, id_test_array = [], []

        # Separate defected and new bearings id
        status = pd.DataFrame([y, groups]).T 
        id_0 = status[status[0] == 0][1].unique()
        id_1 = status[status[0] == 1][1].unique()

        for _ in range(self.n_repeats):
            
            # Stratified slpit into train and test
            id_0_train, id_0_test = id_shuffler(id_0, self.train_size)
            id_1_train, id_1_test = id_shuffler(id_1, self.train_size)

            id_train = np.concatenate([id_0_train, id_1_train])
            id_test = np.concatenate([id_0_test, id_1_test])

            # Indexes that belong to given train and test id
            id_train = status[status[1].map(
                lambda x: x in id_train)].index.to_list()
            id_test = status[status[1].map(
                lambda x: x in id_test)].index.to_list()

            id_train_array.append(id_train), id_test_array.append(id_test)

        # Generator
        for train, test in zip(id_train_array, id_test_array):
            yield train, test
        
        def get_n_splits(self,
                         X: Union[pd.DataFrame, np.ndarray] = None,
                         y: Union[pd.DataFrame, np.ndarray] = None,
                         groups: Union[pd.DataFrame, np.ndarray] = None):
            return self.n_repeats


class PresplitedOverlapGroupCV():

    def __init__(self,
                 subset_size: float = 0.63,
                 n_repeats: int = 100):

        assert 0 <= subset_size <= 1, 'Train size should be between 0 and 1'
        self.subset_size = subset_size
        self.n_repeats = n_repeats

    def split(self,
              X: Union[pd.DataFrame, np.ndarray],
              y: Union[pd.DataFrame, np.ndarray],
              groups: Union[pd.DataFrame, np.ndarray],
              train_groups: Union[pd.DataFrame, np.ndarray],
              test_groups: Union[pd.DataFrame, np.ndarray]) -> Generator:

        id_train_array, id_test_array = [], []

        status = pd.DataFrame([y, groups]).T

        for _ in range(self.n_repeats):
            
            # Reduce size of given train and test groups to necessary
            id_train, _ = id_shuffler(train_groups, self.subset_size)
            id_test, _ = id_shuffler(test_groups, self.subset_size)

            # Indexes that belong to given train and test id
            subset_train_id = status[status[1].map(
                lambda x: x in id_train)].index.to_list()
            subset_test_id = status[status[1].map(
                lambda x: x in id_test)].index.to_list()

            id_train_array.append(subset_train_id), id_test_array.append(subset_test_id)

        # Generator
        for train, test in zip(id_train_array, id_test_array):
            yield train, test
        
        def get_n_splits(self,
                         X: Union[pd.DataFrame, np.ndarray] = None,
                         y: Union[pd.DataFrame, np.ndarray] = None,
                         groups: Union[pd.DataFrame, np.ndarray] = None):
            return self.n_repeats
