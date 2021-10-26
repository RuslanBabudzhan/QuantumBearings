import numpy as np
from sklearn import metrics


class Accuracy:
    @staticmethod
    def count_stat(y_true, y_predict):
        return metrics.accuracy_score(y_true, y_predict)


class Precision:
    @staticmethod
    def count_stat(y_true, y_predict):
        return metrics.precision_score(y_true, y_predict)


class Recall:
    @staticmethod
    def count_stat(y_true, y_predict):
        return metrics.recall_score(y_true, y_predict)


class F1:
    @staticmethod
    def count_stat(y_true, y_predict):
        return metrics.f1_score(y_true, y_predict)


class TPR:
    @staticmethod
    def count_stat(y_true, y_predict):
        TP = sum(np.array(y_true == y_predict == 1))
        P = sum(y_true)
        return TP / P


class TNR:
    @staticmethod
    def count_stat(y_true, y_predict):
        TN = sum(np.array(y_true == y_predict == 0))
        N = len(y_true) - sum(y_true)
        return TN/N
