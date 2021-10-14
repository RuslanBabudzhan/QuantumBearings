from sklearn import metrics


class Accuracy:
    @staticmethod
    def count_stat(data):
        return metrics.accuracy_score(data)


class Precision:
    @staticmethod
    def count_stat(data):
        return metrics.precision_score(data)


class Recall:
    @staticmethod
    def count_stat(data):
        return metrics.recall_score(data)


class F1:
    @staticmethod
    def count_stat(data):
        return metrics.f1_score(data)