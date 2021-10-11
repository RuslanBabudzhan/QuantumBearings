import numpy as np
import scipy.stats as stats

import dataclasses
import json
from typing import List, Union
from enum import Enum

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import feature_selection
from abc import ABC, abstractmethod


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


@dataclasses.dataclass
@abstractmethod
class BaseResultsData(ABC):
    RunLabel: str  # Label assigned to the result
    Model_name: str  # Name of the model (GBM, RF, etc.). Use Model.<model>.n
    Hyperparameters: dict  # dict with hyperparameters names as keys and hyperparameters values as values

    Use_signal: bool  # Was raw signal data used in training
    Use_specter: bool  # Was specter of signal used in training
    Axes: List[str]  # Which axes were used in training. Use Axes.<axis>.name
    Stats: List[str]  # Which statistics were used in training. Use Stats.<stat>.name

    DimReducingMethod: str  # method of dimensionality reduction
    selected_features_ids: List[int]  # indices of features returned by DimReducingMethod and selected for training

    Train_brg_id: List[int]  # Bearing indices used in training
    Test_brg_id: List[int]  # Bearing indices used in testing
    Predictions: List[bool]  # Prediction for each bearing in Test_brg_id

    Accuracy_score: float  # test accuracy score
    Precision_score: float  # test precision score
    Recall_score: float  # test recall score
    F1_score: float  # test f1 score


@dataclasses.dataclass
class SingleRunResults(BaseResultsData):
    pass


@dataclasses.dataclass
class KFoldGridSearchResults:
    HyperparametersGrid: dict  # dict with hyperparameters names as keys and lists of hyperparameters values as values
    cv_count: int  # folds number

    Accuracy_val_score: float  # mean validation accuracy score of best estimator
    Precision_val_score: float  # mean validation precision score of best estimator
    Recall_val_score: float  # mean validation recall score of best estimator
    F1_val_score: float  # mean validation f1 score of best estimator


@dataclasses.dataclass
class BootstrapResults(BaseResultsData):
    Resamplings_number: int  # number of .fit() calls for the model



class Models(Enum):
    """Enumerated estimators"""
    # TODO: Add more models
    GBM = GradientBoostingClassifier
    RF = RandomForestClassifier
    SVC = SVC
    LR = LogisticRegression
    KNN = KNeighborsClassifier


class Axes(Enum):
    """Enumerated axes from raw signals dataset (bearing_signals.csv)"""
    a1_x = 4  # column index from dataset
    a1_y = 5
    a1_z = 6
    a2_x = 7
    a2_y = 8
    a2_z = 9


class Stats(Enum):
    """Enumerated statistics to use in data preparation"""
    # TODO: Add more statistics
    mean: np.mean
    std: np.std
    kurtosis: stats.kurtosis
    skew: stats.skew
    variation: stats.variation


class DimReducers(Enum):
    """Enumerated dimensionality reduces to use before fitting the model"""
    # TODO: Add more reducers
    RFE: feature_selection.RFE


def WriteResultToJSON(result: Union[SingleRunResults, KFoldGridSearchResults, BootstrapResults, list],
                      filename: str,
                      filepath: str = None):
    if filepath:
        fullname = f"{filepath}/{filename}.json"
    else:
        fullname = f"{filename}.json"
    with open(fullname, "w") as write_file:
        if result is list:
            for result_object in result:
                json.dump(result_object, write_file, cls=EnhancedJSONEncoder)
        else:
            json.dump(result, write_file, cls=EnhancedJSONEncoder)
