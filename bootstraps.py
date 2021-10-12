import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from InfFS import select_features

from typing import Tuple, List, Dict
from mljson import SingleRunResults, BootstrapResults, DimReducers, Metrics, WriteResultToJSON
from splits import Splitter


class BootstrapModeler:
    def __init__(self,
                 named_estimators: dict,
                 bearing_indices: np.array = np.arange(1, 113),
                 samples_number: int = 100,
                 train_sample_size: int = 74,
                 feature_dropping_ratio: float = 0.25,
                 should_scale: bool = True,
                 should_reduce_dim: bool = False,
                 reducing_method_name: str = 'RFE',
                 # should_logging: bool = False,
                 logging_type: str = 'silent',
                 leave_positive_features: bool = True,
                 splitter: Splitter = None):
        """
        Class implements bootstrap resampling of test dataset to build a scores distribution for specific models
        """
        self.named_estimators = named_estimators
        self.bearing_indices = bearing_indices
        self.samples_number = samples_number
        self.train_sample_size = train_sample_size
        self.should_scale = should_scale
        self.logging_type = logging_type
        self.should_reduce_dim = should_reduce_dim
        self.feature_dropping_ratio = feature_dropping_ratio
        self.leave_positive_features = leave_positive_features
        self.deleted_features = None
        self.feature_importance = None
        self.whole_experiments_number = 112
        self.splitter = splitter
        self.total_bootstraps_number = 0

        if reducing_method_name not in DimReducers.get_keys():
            raise ValueError(f'bootstrap supports only {DimReducers.get_keys()} dim reducing methods. '
                             f'Got {reducing_method_name}')
        else:
            self.reducing_method = DimReducers[reducing_method_name]

        if logging_type not in Logging.get_keys():
            raise ValueError(f'bootstrap supports only {Logging.get_keys()} log options. '
                             f'Got {logging_type}')
        else:
            self.logging_type = logging_type

        self.should_bootstrap_logging = False
        self.should_separate_logging = False
        if self.logging_type == Logging.bootstrap.name:
            self.should_bootstrap_logging = True
            self.should_separate_logging = False
        elif self.logging_type == Logging.separated.name:
            self.should_bootstrap_logging = False
            self.should_separate_logging = True

        self.should_logging = self.should_bootstrap_logging or self.should_separate_logging

        if self.should_bootstrap_logging:
            self.train_indices = []
            self.test_indices = []

    def run_bootstrap(self, data: pd.DataFrame) -> list:
        if self.should_logging and not isinstance(self.splitter, Splitter):
            raise Exception("splits.Splitter() instance must be attached for logging")

        bootstrap_results = []

        marked_data = self.__mark_data(data)
        stratificated_data = marked_data[marked_data['bearing_id'].isin(self.bearing_indices)]
        # TODO: implement dimension reducing (with selected features indices saving for logging)
        if self.should_reduce_dim:
            reduced_data = self.__reduce_dimensions(stratificated_data)
            data_to_bootstrap = reduced_data
        else:
            data_to_bootstrap = stratificated_data

        for bootstrap_iteration in range(self.samples_number):
            np.random.shuffle(self.bearing_indices)
            self.total_bootstraps_number = bootstrap_iteration
            resampling_results = self.__get_bootstrap_scores(data_to_bootstrap, self.bearing_indices)
            bootstrap_results.append(resampling_results)

        if self.should_bootstrap_logging:
            self.__create_bootstrap_log_file(bootstrap_results.copy())
        return bootstrap_results.copy()

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

        if self.should_bootstrap_logging:
            self.train_indices.append(train_indices.tolist())
            self.test_indices.append(test_indices.tolist())

        for estimator_name, estimator in zip(self.named_estimators.keys(), self.named_estimators.values()):
            estimator.fit(X_train, y_train)
            current_estimator_test_predictions = estimator.predict(X_test)

            current_estimator_results = {
                'accuracy': accuracy_score(y_test, current_estimator_test_predictions),
                'precision': precision_score(y_test, current_estimator_test_predictions),
                'recall': recall_score(y_test, current_estimator_test_predictions),
                'f1': f1_score(y_test, current_estimator_test_predictions),
                'predictions': current_estimator_test_predictions.tolist()
            }
            estimators_data[estimator_name] = current_estimator_results.copy()

            if self.should_separate_logging:
                self.__create_separate_log_file(
                    result=current_estimator_results,
                    model_name=estimator_name,
                    resample_id=self.total_bootstraps_number,
                    train_indices=train_indices.tolist(),
                    test_indices=test_indices.tolist()
                )

        return estimators_data.copy()

    def __mark_data(self, data):
        records_number = len(data)
        records_number_for_bearing = int(records_number / self.whole_experiments_number)
        whole_indices = np.arange(1, self.whole_experiments_number + 1)
        records_labels = np.repeat(whole_indices, records_number_for_bearing)
        marked_data = data.copy()
        marked_data['bearing_id'] = records_labels.tolist()
        return marked_data

    def __reduce_dimensions(self, stratificated_data):
        columns = [str(col) for col in range(stratificated_data.shape[1])]
        named_data = stratificated_data.drop(labels='bearing_id', axis=1).copy()
        named_data.columns = columns
        features_importance = select_features(named_data.iloc[:, 1:], named_data.iloc[:, 0], 0.5)
        if self.leave_positive_features:
            features_to_drop = features_importance[features_importance['importance'] < 0]['features'].to_numpy()
        else:
            features_to_drop_number = int(len(features_importance) * self.feature_dropping_ratio)
            assert features_to_drop_number != len(features_importance), "all features must be dropped, " \
                                                                        "reduce feature_dropping_ratio"
            features_to_drop = features_importance[:features_to_drop_number]['features'].to_numpy()
        self.deleted_features = features_to_drop
        self.feature_importance = features_importance
        reduced_data = named_data.drop(labels=features_to_drop, axis=1)
        reduced_data.columns = [int(col_str) for col_str in reduced_data.columns]
        reduced_data['bearing_id'] = stratificated_data['bearing_id']
        return reduced_data

    def __create_separate_log_file(self, result, model_name, resample_id,
                                   train_indices, test_indices, result_label='test'):
        results_logger = SingleRunResults(
            run_label=result_label,
            model_name=model_name,
            hyperparameters=self.named_estimators[model_name].get_params(),

            use_signal=self.splitter.use_signal,
            use_specter=self.splitter.use_specter,
            axes=self.splitter.frequency_data_columns,
            stats=list(self.splitter.stats.keys()),

            dim_reducing_method=self.reducing_method.name,
            # selected_features_ids = self.selected_features_ids
            selected_features_ids=[],

            train_brg_id=train_indices,
            test_brg_id=test_indices,
            predictions=result['predictions'],

            accuracy_score=result['accuracy'],
            precision_score=result['precision'],
            recall_score=result['recall'],
            f1_score=result['f1'],

            resampling_number=self.samples_number
            # TODO: replace selected_features_ids field with proper data
        )
        WriteResultToJSON(results_logger.dict(),
                          f"{result_label}_bootstrap_{model_name}_{resample_id}",
                          "SimpleRunsJSONs")

    def __create_bootstrap_log_file(self, results, result_label='test'):
        models_names = self.named_estimators.keys()
        resampling_results_names = ['accuracy', 'precision', 'recall', 'f1', 'predictions']
        for model_name in models_names:
            model_scores = {}
            for result_name in resampling_results_names:
                model_score = [result[model_name][result_name] for result in results]
                model_scores[result_name] = model_score.copy()
            results_logger = BootstrapResults(
                run_label=result_label,
                model_name=model_name,
                hyperparameters=self.named_estimators[model_name].get_params(),

                use_signal=self.splitter.use_signal,
                use_specter=self.splitter.use_specter,
                axes=self.splitter.frequency_data_columns,
                stats=list(self.splitter.stats.keys()),

                dim_reducing_method=self.reducing_method.name,
                # selected_features_ids = self.selected_features_ids
                selected_features_ids=[],

                train_brg_id=self.train_indices,
                test_brg_id=self.test_indices,
                predictions=model_scores['predictions'],

                accuracy_score=model_scores['accuracy'],
                precision_score=model_scores['precision'],
                recall_score=model_scores['recall'],
                f1_score=model_scores['f1'],

                resampling_number=self.samples_number
                # TODO: replace selected_features_ids field with proper data
            )
            WriteResultToJSON(results_logger.dict(), f"{result_label}_bootstrap_{model_name}", "BootstrapJSONs")

    def plot_bootstrap_results(self, results, plot_subtitle=None, verbose=True):
        plt.figure(figsize=(11, 6), dpi=80)
        palette = iter(sns.husl_palette(len(self.named_estimators)))
        alpha = 0.95
        for estimator_name in list(self.named_estimators.keys()):
            estimator_scores = [resampling_result[estimator_name]['f1'] for resampling_result in results]
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


class Logging(Enum):
    """logging degree"""
    silent = 1
    bootstrap = 2
    separated = 3

    @staticmethod
    def get_keys() -> List[str]:
        return list(map(lambda c: c.name, Logging))

