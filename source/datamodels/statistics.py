import numpy as np
from scipy import stats
from pyentrp import entropy as pent
import antropy as ent
from hurst import compute_Hc

"""
This file implements statistics for features generation
"""


class Mean:  # statistic №1
    @staticmethod
    def count_stat(data):
        return np.mean(data)


class STD:  # statistic №2
    @staticmethod
    def count_stat(data):
        return np.std(data)


class Kurtosis:  # statistic №3
    @staticmethod
    def count_stat(data):
        return stats.kurtosis(data)


class Skew:  # statistic №4
    @staticmethod
    def count_stat(data):
        return stats.skew(data)


class Variation:  # statistic №5
    @staticmethod
    def count_stat(data):
        return stats.variation(data)


class StatRange:  # statistic №6
    @staticmethod
    def count_stat(data):
        return max(data) - min(data)


class IQR:  # statistic №7
    @staticmethod
    def count_stat(data):
        return stats.iqr(data)


class SampleEntropy:  # statistic №8
    @staticmethod
    def count_stat(data):
        return float(pent.sample_entropy(data, 1))


class ShannonEntropy:  # statistic №9
    @staticmethod
    def count_stat(data):
        return pent.shannon_entropy(data)


class Energy:  # statistic №10
    @staticmethod
    def count_stat(data):
        return sum(np.abs(data) ** 2)


class Hurst:  # statistic №11
    @staticmethod
    def count_stat(data):
        h, _, _ = compute_Hc(data, kind='change')
        return h


class PetrosianFD:  # statistic №12
    @staticmethod
    def count_stat(data):
        return ent.petrosian_fd(data)


class ZeroCrossing:  # statistic №13
    @staticmethod
    def count_stat(data):
        return ent.num_zerocross(data)


class HiguchiFD:  # statistic №14
    @staticmethod
    def count_stat(data):
        return ent.higuchi_fd(data)


class Activity:  # statistic №15
    @staticmethod
    def count_stat(data):
        activity, _ = ent.hjorth_params(data)
        return activity


class Complexity:  # statistic №16
    @staticmethod
    def count_stat(data):
        _, complexity = ent.hjorth_params(data)
        return complexity


class CrestFactor:  # statistic №17
    @staticmethod
    def count_stat(data):
        return np.max(np.abs(data)) / np.sqrt(np.mean(np.square(data)))
