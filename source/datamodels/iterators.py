from typing import List
from enum import Enum

from sklearn.metrics import make_scorer

from source.datamodels import metrics, statistics


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
    permutation_entropy = statistics.PermutationEntropy
    svd_entropy = statistics.SVDEntropy
    approx_entropy = statistics.ApproximateEntropy
    katz_fd = statistics.KatzFD
    detrended_fluctuation = statistics.DetrendedFluctuationAnalysis

    @staticmethod
    def get_keys() -> List[str]:
        return list(map(lambda c: c.name, Stats))


class Metrics(Enum):
    """
    Enumerated metrics

    Uses only metrics.py implementations of metrics
    """
    accuracy = metrics.Accuracy
    precision = metrics.Precision
    recall = metrics.Recall
    f1 = metrics.F1
    TPR = metrics.TPR
    TNR = metrics.TNR

    @staticmethod
    def get_keys() -> List[str]:
        return list(map(lambda c: c.name, Metrics))

    @staticmethod
    def get_scorers_dict() -> dict:
        """
        scores dict needs to use in cross_validate
        :return: dict of score names as keys and scorers functions as values
        """
        scores_dict = dict()
        for metric in Metrics:
            scores_dict[metric.name] = make_scorer(metric.value.score_func)
        return scores_dict
