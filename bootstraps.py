import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from InfFS import select_features


class BootstrapModeler:
    def __init__(self, named_estimators, bearing_indices, samples_number=100, train_sample_size=74,
                 feature_dropping_ratio=0.25, should_scale=True, should_reduce_dim=False, should_logging=False,
                 leave_positive_features=True):
        """Constructor"""
        self.named_estimators = named_estimators
        self.bearing_indices = bearing_indices
        self.samples_number = samples_number
        self.train_sample_size = train_sample_size
        self.should_scale = should_scale
        self.should_logging = should_logging
        self.should_reduce_dim = should_reduce_dim
        self.feature_dropping_ratio = feature_dropping_ratio
        self.leave_positive_features = leave_positive_features
        self.deleted_features = None
        self.feature_importance = None
        self.whole_experiments_number = 112

    def run_bootstrap(self, data):
        bootstrap_results = []
        marked_data = self.__mark_data(data)
        stratificated_data = marked_data[marked_data['bearing_id'].isin(self.bearing_indices)]
        # print(len(self.bearing_indices))
        # print(stratificated_data.shape)
        if self.should_reduce_dim:
            columns = [str(col) for col in range(data.shape[1])]
            named_data = stratificated_data.drop(labels='bearing_id', axis=1).copy()
            named_data.columns = columns
            # print(named_data.shape)
            features_importance = select_features(named_data.iloc[:, 1:], named_data.iloc[:, 0], 0.5)
            if self.leave_positive_features:
                features_to_drop = features_importance[features_importance['importance'] < 0]['features'].to_numpy()
            else:
                features_to_drop_number = int(len(features_importance)*self.feature_dropping_ratio)
                assert features_to_drop_number != len(features_importance), "all features must be dropped, " \
                                                                            "reduce feature_dropping_ratio"
                features_to_drop = features_importance[:features_to_drop_number]['features'].to_numpy()
            self.deleted_features = features_to_drop
            self.feature_importance = features_importance
            reduced_data = named_data.drop(labels=features_to_drop, axis=1)
            reduced_data.columns = [int(col_str) for col_str in reduced_data.columns]
            reduced_data['bearing_id'] = stratificated_data['bearing_id']
            data_to_bootstrap = reduced_data
            # print(data_to_bootstrap.shape)
            # print(data_to_bootstrap.shape)
            # marked_data = self.__mark_data(reduced_data)
        else:
            data_to_bootstrap = stratificated_data
            # marked_data = self.__mark_data(data)

        for bootstrap_iteration in range(self.samples_number):
            np.random.shuffle(self.bearing_indices)
            resampling_results = self.__get_bootstrap_scores(data_to_bootstrap, self.bearing_indices)
            bootstrap_results.append(resampling_results)
        return bootstrap_results

    def __mark_data(self, data):
        records_number = len(data)
        records_number_for_bearing = int(records_number/self.whole_experiments_number)
        whole_indices = np.arange(1, self.whole_experiments_number+1)
        records_labels = np.repeat(whole_indices, records_number_for_bearing)
        marked_data = data.copy()
        marked_data['bearing_id'] = records_labels.tolist()
        return marked_data

    def __get_bootstrap_scores(self, data, shuffled_indices):
        train_indices = shuffled_indices[:self.train_sample_size]
        test_indices = shuffled_indices[self.train_sample_size:]

        records_train = data[data['bearing_id'].isin(train_indices)]
        records_test = data[data['bearing_id'].isin(test_indices)]

        records_train = records_train.to_numpy()
        records_test = records_test.to_numpy()

        X_train = records_train[:, 1:records_train.shape[1] - 1]
        y_train = records_train[:, 0]
        X_test = records_test[:, 1:records_test.shape[1] - 1]
        y_test = records_test[:, 0]

        if self.should_scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        estimators_data = {}

        for estimator_name, estimator in zip(self.named_estimators.keys(), self.named_estimators.values()):
            estimator.fit(X_train, y_train)
            current_estimator_test_predictions = estimator.predict(X_test)
            base_estimator_test_score = f1_score(y_test, current_estimator_test_predictions)
            current_estimator_results = {'score': base_estimator_test_score}
            if self.should_logging:
                current_estimator_results['train_indices'] = train_indices.copy()
                current_estimator_results['test_indices'] = test_indices.copy()
            estimators_data[estimator_name] = current_estimator_results.copy()

        return estimators_data.copy()

    def plot_bootstrap_results(self, results, plot_subtitle=None, verbose=True):
        plt.figure(figsize=(11, 6), dpi=80)
        palette = iter(sns.husl_palette(len(self.named_estimators)))
        alpha = 0.95
        for estimator_name in list(self.named_estimators.keys()):
            estimator_scores = [resampling_result[estimator_name]['score'] for resampling_result in results]
            if verbose:
                print(f'mean base estimator ({estimator_name}) score: {np.round(np.mean(estimator_scores), 3)}')
            sns.kdeplot(estimator_scores, alpha=alpha, label=f'{estimator_name} score', color=next(palette))

        plt.xlabel('F1 score')
        if plot_subtitle is not None:
            plt.title('F1 score distribution. ' + plot_subtitle + '.')
        else:
            plt.title('F1 score distribution')
        plt.legend()
        plt.show()
