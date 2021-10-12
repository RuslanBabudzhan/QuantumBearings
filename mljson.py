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
import sklearn.metrics as metrics
from abc import ABC, abstractmethod

import statistics


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


@dataclasses.dataclass
@abstractmethod
class BaseResultsData(ABC):
    run_label: str  # Label assigned to the result
    model_name: str  # Name of the model (GBM, RF, etc.). Use Model.<model>.n
    hyperparameters: dict  # dict with hyperparameters names as keys and hyperparameters values as values

    use_signal: bool  # Was raw signal data used in training
    use_specter: bool  # Was specter of signal used in training
    axes: List[str]  # Which axes were used in training. Use Axes.<axis>.name
    stats: List[str]  # Which statistics were used in training. Use Stats.<stat>.name

    dim_reducing_method: str  # method of dimensionality reduction
    selected_features_ids: List[int]  # indices of features returned by DimReducingMethod and selected for training

    train_brg_id: List[int]  # Bearing indices used in training
    test_brg_id: List[int]  # Bearing indices used in testing
    predictions: List[float]  # Prediction for each bearing in Test_brg_id

    accuracy_score: float  # test accuracy score
    precision_score: float  # test precision score
    recall_score: float  # test recall score
    f1_score: float  # test f1 score


@dataclasses.dataclass
class SingleRunResults(BaseResultsData):
    pass


@dataclasses.dataclass
class KFoldGridSearchResults:
    hyperparameters_grid: dict  # dict with hyperparameters names as keys and lists of hyperparameters values as values
    cv_count: int  # folds number

    accuracy_val_score: float  # mean validation accuracy score of best estimator
    precision_val_score: float  # mean validation precision score of best estimator
    recall_val_score: float  # mean validation recall score of best estimator
    f1_val_score: float  # mean validation f1 score of best estimator


@dataclasses.dataclass
class BootstrapResults(BaseResultsData):
    resampling_number: int  # number of .fit() calls for the model

    # use_signal: bool  # Was raw signal data used in training
    # use_specter: bool  # Was specter of signal used in training
    # axes: List[str]  # Which axes was used in training. Use Axes.<axis>.name
    # stats: List[str]  # Which axes was used in training. Use Stats.<axis>.name

    train_brg_id: List[List[int]]  # Bearing indices used in training for each resampling
    test_brg_id: List[List[int]]  # Bearing indices used in testing for each resampling
    predictions: List[List[bool]]  # Prediction for each bearing in Test_brg_id

    accuracy_score: List[float]  # list of test accuracy score for each resampling
    precision_score: List[float]  # test precision score for each resampling
    recall_score: List[float]  # test recall score for each resampling
    f1_score: List[float]  # test f1 score for each resampling


class Models(Enum):
    """Enumerated estimators"""
    # TODO: Add more models
    GBM = GradientBoostingClassifier
    RF = RandomForestClassifier
    SVC = SVC
    LR = LogisticRegression
    KNN = KNeighborsClassifier

    @staticmethod
    def get_keys() -> List[str]:
        return list(map(lambda c: c.name, Models))


class Axes(Enum):
    """Enumerated axes from raw signals dataset (bearing_signals.csv)"""
    a1_x = 4  # column index from dataset
    a1_y = 5
    a1_z = 6
    a2_x = 7
    a2_y = 8
    a2_z = 9

    @staticmethod
    def get_keys() -> List[str]:
        return list(map(lambda c: c.name, Axes))


class Stats(Enum):
    """Enumerated statistics to use in data preparation"""
    # TODO: Add more statistics
    mean = np.mean
    std = np.std
    kurtosis = stats.kurtosis
    skew = stats.skew
    variation = stats.variation
    range = statistics.stat_range
    iqr = stats.iqr
    sample_entropy = statistics.sample_entropy
    shannon_entropy = statistics.shannon_entropy
    energy = statistics.energy
    hurst = statistics.hurst
    petrosian_fd = statistics.petrosian_fd
    zero_crossing = statistics.zero_crossing
    higuchi_fd = statistics.higuchi_fd
    activity = statistics.activity
    complexity = statistics.complexity
    crest_factor = statistics.crest_factor

    @staticmethod
    def get_keys() -> List[str]:
        return list(map(lambda c: c.name, Stats))

    # @staticmethod
    # def get_keys() -> List[str]:
    #     return [el.name for el in Stats]


class DimReducers(Enum):
    """Enumerated dimensionality reduces to use before fitting the model"""
    # TODO: Add more reducers
    RFE = feature_selection.RFE

    @staticmethod
    def get_keys() -> List[str]:
        return list(map(lambda c: c.name, DimReducers))


class Metrics(Enum):
    """Enumerated metrics"""
    accuracy = metrics.accuracy_score
    precision = metrics.precision_score
    recall = metrics.recall_score
    f1 = metrics.f1_score

    @staticmethod
    def get_keys() -> List[str]:
        return list(map(lambda c: c.name, Metrics))


def WriteResultToJSON(result: Union[BaseResultsData, List[BaseResultsData]],
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
