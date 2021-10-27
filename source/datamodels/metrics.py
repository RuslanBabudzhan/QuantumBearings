import numpy as np
from sklearn import metrics


class Accuracy:
    @staticmethod
    def score_func(y, y_pred):
        return metrics.accuracy_score(y, y_pred)


class Precision:
    @staticmethod
    def score_func(y, y_pred):
        return metrics.precision_score(y, y_pred)


class Recall:
    @staticmethod
    def score_func(y, y_pred):
        return metrics.recall_score(y, y_pred)


class F1:
    @staticmethod
    def score_func(y, y_pred):
        return metrics.f1_score(y, y_pred)


class TPR:
    @staticmethod
    def score_func(y, y_pred):
        TP = np.sum(np.logical_and(np.equal(y, y_pred), np.equal(y, 1)).astype(int))
        P = sum(y)
        return TP / P


class TNR:
    @staticmethod
    def score_func(y, y_pred):
        TN = np.sum(np.logical_and(np.equal(y, y_pred), np.equal(y, 0)).astype(int))
        N = len(y) - sum(y)
        return TN/N
