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
    def count_stat(data, axis=1):
        return np.mean(data, axis=axis)


class STD:  # statistic №2
    @staticmethod
    def count_stat(data, axis=1):
        return np.std(data, axis=axis)


class Kurtosis:  # statistic №3
    @staticmethod
    def count_stat(data, axis=1):
        return stats.kurtosis(data, axis=axis)


class Skew:  # statistic №4
    @staticmethod
    def count_stat(data, axis=1):
        return stats.skew(data, axis=axis)


class Variation:  # statistic №5
    @staticmethod
    def count_stat(data, axis=1):
        return stats.variation(data, axis=axis)


class StatRange:  # statistic №6
    @staticmethod
    def count_stat(data, axis=1):
        return np.max(data, axis=axis) - np.min(data, axis=axis)


class IQR:  # statistic №7
    @staticmethod
    def count_stat(data, axis=1):
        return stats.iqr(data, axis=axis)


class SampleEntropy:  # statistic №8
    @staticmethod
    def count_stat(data, axis=1):
        rows_count = data.shape[0]
        ent_list = []
        for row in range(rows_count):
            if axis == 1:
                data_row = data[row, :]
            else:
                data_row = data[:, row]
            entr = float(pent.sample_entropy(data_row, 1))
            ent_list.append(entr)
        return np.array(ent_list)


class ShannonEntropy:  # statistic №9
    @staticmethod
    def count_stat(data, axis=1):
        rows_count = data.shape[0]
        ent_list = []
        for row in range(rows_count):
            if axis == 1:
                data_row = data[row, :]
            else:
                data_row = data[:, row]
            entr = pent.shannon_entropy(data_row)
            ent_list.append(entr)
        return np.array(ent_list)


class Energy:  # statistic №10
    @staticmethod
    def count_stat(data, axis=1):
        return np.sum(np.power(np.abs(data), 2), axis=axis)


class Hurst:  # statistic №11
    @staticmethod
    def count_stat(data, axis=1):
        rows_count = data.shape[0]
        h_list = []
        for row in range(rows_count):
            if axis == 1:
                data_row = data[row, :]
            else:
                data_row = data[:, row]
            h, _, _ = compute_Hc(data_row, kind='change')
            h_list.append(h)
        return np.array(h_list)


class PetrosianFD:  # statistic №12
    @staticmethod
    def count_stat(data, axis=1):
        return ent.petrosian_fd(data, axis=axis)


class ZeroCrossing:  # statistic №13
    @staticmethod
    def count_stat(data, axis=1):
        return ent.num_zerocross(data, axis=axis)


class HiguchiFD:  # statistic №14
    @staticmethod
    def count_stat(data, axis=1):
        rows_count = data.shape[0]
        h_list = []
        for row in range(rows_count):
            if axis == 1:
                data_row = data[row, :]
            else:
                data_row = data[:, row]
            h = ent.higuchi_fd(data_row)
            h_list.append(h)
        return np.array(h_list)


class Activity:  # statistic №15
    @staticmethod
    def count_stat(data, axis=1):
        rows_count = data.shape[0]
        h_list = []
        for row in range(rows_count):
            if axis == 1:
                data_row = data[row, :]
            else:
                data_row = data[:, row]
            activity, _ = ent.hjorth_params(data_row)
            h_list.append(activity)
        return np.array(h_list)


class Complexity:  # statistic №16
    @staticmethod
    def count_stat(data, axis=1):
        rows_count = data.shape[0]
        h_list = []
        for row in range(rows_count):
            if axis == 1:
                data_row = data[row, :]
            else:
                data_row = data[:, row]
            _, complexity = ent.hjorth_params(data_row)
            h_list.append(complexity)
        return np.array(h_list)


class CrestFactor:  # statistic №17
    @staticmethod
    def count_stat(data, axis=1):
        mx = np.max(np.abs(data), axis=axis)
        sq = np.sqrt(np.mean(np.square(data), axis=axis))
        return mx/sq


class PermutationEntropy:  # statistic №18
    @staticmethod
    def count_stat(data, axis=1):
        rows_count = data.shape[0]
        perm_entropy = []
        for row in range(rows_count):
            if axis == 1:
                data_row = data[row, :]
            else:
                data_row = data[:, row]
            perm_entropy.append(ent.perm_entropy(data_row))
    
        return np.array(perm_entropy)


class SVDEntropy:  # statistic №19
    @staticmethod
    def count_stat(data, axis=1):
        rows_count = data.shape[0]
        entropy_list = []
        for row in range(rows_count):
            if axis == 1:
                data_row = data[row, :]
            else:
                data_row = data[:, row]
            entropy_list.append(ent.svd_entropy(data_row, normalize=True))
        return entropy_list


class ApproximateEntropy:  # statistic №20
    @staticmethod
    def count_stat(data, axis=1):
        rows_count = data.shape[0]
        entropy_list = []
        for row in range(rows_count):
            if axis == 1:
                data_row = data[row, :]
            else:
                data_row = data[:, row]
            entropy_list.append(ent.app_entropy(data_row))
        return entropy_list


class KatzFD:  # statistic №21
    @staticmethod
    def count_stat(data, axis=1):
        return ent.katz_fd(data, axis=axis)


class DetrendedFluctuationAnalysis:  # statistic №22
    @staticmethod
    def count_stat(data, axis=1):
        rows_count = data.shape[0]
        DFA_list = []
        for row in range(rows_count):
            if axis == 1:
                data_row = data[row, :]
            else:
                data_row = data[:, row]
            DFA_list.append(ent.detrended_fluctuation(data_row))
        return DFA_list
