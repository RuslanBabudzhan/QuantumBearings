import numpy as np
import pandas as pd
from typing import Union, List
from abc import ABC, abstractmethod


class Shuffler(ABC):

    @abstractmethod
    def split(self):
        pass
    
    def __init__(self, 
                 train_size: float, 
                 n_repeats: int = 100):

        assert 0 <= train_size <= 1, 'Train size should be between 0 and 1' 
        self.train_size = train_size
        self.n_repeats = n_repeats


    def id_shuffler(self, 
                    labels: Union[pd.DataFrame, np.ndarray], 
                    train_size: float) -> np.ndarray:
        
        len_size = len(labels)
        N = int(len_size * train_size)
        labels = np.array(labels)
        np.random.shuffle(labels)
        train, test = labels[:N], labels[N:]

        return train, test
        

class OverlapGroupCV(Shuffler):

    def split(self, 
              id_0: Union[pd.DataFrame, np.ndarray], 
              id_1: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:

        # id_train_array, id_test_array = [], []

        for _ in range(self.n_repeats):

            id_0_train, id_0_test = self.id_shuffler(id_0, self.train_size)
            id_1_train, id_1_test = self.id_shuffler(id_1, self.train_size)

            id_train = np.concatenate([id_0_train, id_1_train])
            id_test = np.concatenate([id_0_test, id_1_test])

            # id_train_array.append(id_train), id_test_array.append(id_test)
            
            yield id_train, id_test
    

class PresplitedOverlapGroupCV(Shuffler):

    def split(self,
               labels: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        
        # id_sub_array= []
        
        for _ in range(self.n_repeats):

            id_sub, _ = self.id_shuffler(labels, self.train_size)
            # id_sub_array.append(id_sub)

            yield id_sub


data_labels = pd.read_csv('Data/bearing_classes.csv', sep=";") # load bearings labeles

labels_big_0 = data_labels.iloc[1:31] # old big bearings id
labels_big_1 = data_labels.iloc[101:108] # new big bearings id

labels_small_0 = data_labels.iloc[31:101] # old small bearings id
labels_small_1 = data_labels.iloc[108:] # new small bearings id

bear_id_0 = pd.concat([labels_big_0, labels_small_0]) # all old bearings id
bear_id_1 = pd.concat([labels_big_1, labels_small_1]) # all new bearings id

shuffle_1 = OverlapGroupCV(0.8, n_repeats=1)
shuffle_2 = PresplitedOverlapGroupCV(0.8, n_repeats=1)

bear_id_train, bear_id_test = shuffle_1.split(bear_id_0.bearing_id, bear_id_1.bearing_id) # get 100 train\test id splits
subset = shuffle_1.split(bear_id_0.bearing_id) # get 100 different id subsets