from typing import List
from enum import Enum
from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import feature_selection

import metrics
import statistics


@abstractmethod
class BaseResultsData(ABC, BaseModel):
    """ Class implements abstract model of data obtained as a result of ML algorithm launch """
    run_label: str  # Label assigned to the result
    model_name: str  # Name of the model (GBM, RF, etc.). Use Model.<model>.n
    hyperparameters: dict  # dict with hyperparameters names as keys and hyperparameters values as values

    use_signal: bool  # Was raw signal data used in training
    use_specter: bool  # Was specter of signal used in training
    axes: List[str]  # Which axes were used in training. Use Axes.<axis>.name
    stats: List[str]  # Which statistics were used in training. Use Stats.<stat>.name

    dim_reducing: bool  # was dim reducing used before training
    dim_reducing_method: str  # method of dimensionality reduction
    selected_features_ids: List[int]  # indices of features returned by DimReducingMethod and selected for training

    train_brg_id: List[int]  # Bearing indices used in training
    test_brg_id: List[int]  # Bearing indices used in testing
    predictions: List[float]  # Prediction for each bearing in Test_brg_id

    accuracy_score: float  # test accuracy score
    precision_score: float  # test precision score
    recall_score: float  # test recall score
    f1_score: float  # test f1 score

    @abstractmethod
    def __str__(self):
        """implementation of human-readable representation of ResultsData objects"""
        return


class SingleRunResults(BaseResultsData):
    """ Model of data obtained as a result of single ML algorithm launch """
    def __str__(self):
        """implementation of human-readable representation of the object"""
        str_representation = f"Result name: {self.run_label}\n" \
                             f"ML model: {self.model_name}\n" \
                             f"Hyperparameters: {self.hyperparameters}\n" \
                             f"Trained on signal data: {self.use_signal}\n" \
                             f"Trained on specter data: {self.use_specter}\n" \
                             f"Bearing signal axes: {self.axes}\n" \
                             f"Statistics for features generation: {self.stats}\n" \
                             f"Method of dimensionality reducing: {self.dim_reducing_method}\n" \
                             f"Scores: accuracy = {self.accuracy_score:.3f}, precision = {self.precision_score:.3f}, " \
                             f"recall = {self.recall_score:.3f}, F1 = {self.f1_score:.3f}"
        return str_representation


class KFoldGridSearchResults(BaseResultsData):
    """ Model of data obtained as a result of KFold model tuning """
    hyperparameters_grid: dict  # dict with hyperparameters names as keys and lists of hyperparameters values as values
    cv_count: int  # folds number

    accuracy_val_score: float  # mean validation accuracy score of best estimator
    precision_val_score: float  # mean validation precision score of best estimator
    recall_val_score: float  # mean validation recall score of best estimator
    f1_val_score: float  # mean validation f1 score of best estimator

    def __str__(self):
        """implementation of human-readable representation of the object"""
        str_representation = f"Result name: {self.run_label}\n" \
                             f"ML model: {self.model_name}\n" \
                             f"Count of folds: {self.cv_count}\n" \
                             f"Hyperparameters of best model: {self.hyperparameters}\n" \
                             f"Trained on signal data: {self.use_signal}\n" \
                             f"Trained on specter data: {self.use_specter}\n" \
                             f"Bearing signal axes: {self.axes}\n" \
                             f"Statistics for features generation: {self.stats}\n" \
                             f"Method of dimensionality reducing: {self.dim_reducing_method}\n" \
                             f"Scores: accuracy = {self.accuracy_score:.3f}, precision = {self.precision_score:.3f}, " \
                             f"recall = {self.recall_score:.3f}, F1 = {self.f1_score:.3f}"
        return str_representation


class BootstrapResults(BaseResultsData):
    """ Model of data obtained as a result of Bootstrap over single ML algorithm """
    resampling_number: int  # number of .fit() calls for the model

    train_brg_id: List[List[int]]  # Bearing indices used in training for each resampling
    test_brg_id: List[List[int]]  # Bearing indices used in testing for each resampling
    predictions: List[List[bool]]  # Prediction for each bearing in Test_brg_id

    accuracy_score: List[float]  # list of test accuracy score for each resampling
    precision_score: List[float]  # test precision score for each resampling
    recall_score: List[float]  # test recall score for each resampling
    f1_score: List[float]  # test f1 score for each resampling

    def __str__(self):
        """implementation of human-readable representation of the object"""
        str_representation = f"Result name: {self.run_label}\n" \
                             f"ML model: {self.model_name}\n" \
                             f"Count of resamplings: {self.resampling_number}\n" \
                             f"Hyperparameters: {self.hyperparameters}\n" \
                             f"Trained on signal data: {self.use_signal}\n" \
                             f"Trained on specter data: {self.use_specter}\n" \
                             f"Bearing signal axes: {self.axes}\n" \
                             f"Statistics for features generation: {self.stats}\n" \
                             f"Method of dimensionality reducing: {self.dim_reducing_method}\n" \
                             f"Mean scores: accuracy = {np.mean(self.accuracy_score):.3f}, " \
                             f"precision = {np.mean(self.precision_score):.3f}, " \
                             f"recall = {np.mean(self.recall_score):.3f}, " \
                             f"F1 = {np.mean(self.f1_score):.3f}"
        return str_representation


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
    """
    Enumerated statistics to use in data preparation.

    Uses only statistics.py implementations of statistics
    """
    mean = statistics.Mean
    std = statistics.STD
    kurtosis = statistics.Kurtosis
    skew = statistics.Skew
    variation = statistics.Variation
    range = statistics.StatRange
    iqr = statistics.IQR
    sample_entropy = statistics.SampleEntropy
    shannon_entropy = statistics.ShannonEntropy
    energy = statistics.Energy
    hurst = statistics.Hurst
    petrosian_fd = statistics.PetrosianFD
    zero_crossing = statistics.ZeroCrossing
    higuchi_fd = statistics.HiguchiFD
    activity = statistics.Activity
    complexity = statistics.Complexity
    crest_factor = statistics.CrestFactor

    @staticmethod
    def get_keys() -> List[str]:
        return list(map(lambda c: c.name, Stats))


class DimReducers(Enum):
    """Enumerated dimensionality reduces to use before fitting the model"""
    # TODO: Add more reducers
    RFE = feature_selection.RFE

    @staticmethod
    def get_keys() -> List[str]:
        return list(map(lambda c: c.name, DimReducers))


class Metrics(Enum):
    """Enumerated metrics"""
    accuracy = metrics.Accuracy
    precision = metrics.Precision
    recall = metrics.Recall
    f1 = metrics.F1

    @staticmethod
    def get_keys() -> List[str]:
        return list(map(lambda c: c.name, Metrics))
