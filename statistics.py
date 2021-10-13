from pyentrp import entropy as pent
import antropy as ent
from hurst import compute_Hc
import numpy as np
from scipy import stats

"""
This file implements statistics for features generation
"""


class Mean:
    @staticmethod
    def count_stat(data):
        return np.mean(data)


class STD:
    @staticmethod
    def count_stat(data):
        return np.std(data)


class Kurtosis:
    @staticmethod
    def count_stat(data):
        return stats.kurtosis(data)


class Skew:
    @staticmethod
    def count_stat(data):
        return stats.skew(data)


class Variation:
    @staticmethod
    def count_stat(data):
        return stats.variation(data)


class StatRange:
    @staticmethod
    def count_stat(data):
        return max(data) - min(data)


class IQR:
    @staticmethod
    def count_stat(data):
        return stats.iqr(data)


class SampleEntropy:
    @staticmethod
    def count_stat(data):
        return float(pent.sample_entropy(data, 1))


class ShannonEntropy:
    @staticmethod
    def count_stat(data):
        return pent.shannon_entropy(data)


class Energy:
    @staticmethod
    def count_stat(data):
        return sum(np.abs(data) ** 2)


class Hurst:
    @staticmethod
    def count_stat(data):
        h, _, _ = compute_Hc(data, kind='change')
        return h


class PetrosianFD:
    @staticmethod
    def count_stat(data):
        return ent.petrosian_fd(data)


class ZeroCrossing:
    @staticmethod
    def count_stat(data):
        return ent.num_zerocross(data)


class HiguchiFD:
    @staticmethod
    def count_stat(data):
        return ent.higuchi_fd(data)


class Activity:
    @staticmethod
    def count_stat(data):
        activity, _ = ent.hjorth_params(data)
        return activity


class Complexity:
    @staticmethod
    def count_stat(data):
        _, complexity = ent.hjorth_params(data)
        return complexity


class CrestFactor:
    @staticmethod
    def count_stat(data):
        return np.max(np.abs(data)) / np.sqrt(np.mean(np.square(data)))
