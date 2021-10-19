from typing import Union, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from source.processes.base import BaseModeler, Logging
from source.postprocessing.mljson import write_result_obj_to_json
from source.datamodels.datamodels import SingleRunResults, BootstrapResults, DimReducers
from source.preprocessing.splits import Splitter


# from source.preprocessing.reduce import Reducer # TODO implement feature reducing


class BootstrapModeler(BaseModeler):
    """
    Class implements bootstrap resampling of test dataset to build a scores distribution for specific models
    """

    def __init__(self,
                 named_estimators: dict,
                 samples_number: int = 100,
                 test_sample_size: int = 33,
                 feature_dropping_ratio: float = 0.25,
                 should_scale: bool = True,
                 should_reduce_dim: bool = False,
                 reducing_method_name: str = 'RFE',
                 splitter: Splitter = None,
                 # reducer: Reducer = None, # TODO: Add reducer
                 ):

        self.logging_type = 'separated'
        self._should_logging = False
        self.__should_bootstrap_logging = False
        self.__should_separate_logging = False
        self._log_folder = None
        self.__result_label_prefix = None

        self.named_estimators = named_estimators
        self.samples_number = samples_number
        self.test_sample_size = test_sample_size
        self.should_scale = should_scale
        self.should_reduce_dim = should_reduce_dim
        self.feature_dropping_ratio = feature_dropping_ratio
        self.splitter = splitter

        if reducing_method_name not in DimReducers.get_keys():
            raise ValueError(f'bootstrap supports only {DimReducers.get_keys()} dim reducing methods. '
                             f'Got {reducing_method_name}')
        else:
            self.reducing_method = DimReducers[reducing_method_name]

    def initialize_logging(self,
                           result_label_prefix: str,
                           logging_type: str = 'bootstrap',
                           log_folder: Optional[str] = None,
                           ):
        self.__result_label_prefix = result_label_prefix

        if logging_type not in Logging.get_keys():
            raise ValueError(f'bootstrap supports only {Logging.get_keys()} log options. '
                             f'Got {logging_type}')
        else:
            self.logging_type = logging_type

        self.__should_bootstrap_logging = False
        self.__should_separate_logging = False

        if self.logging_type == Logging.experiment.name:
            self.__should_bootstrap_logging = True
            self.__should_separate_logging = False
            self._log_folder = log_folder or 'F:/PythonNotebooks/Study/Quantum/Bearings/experiments/Bootstraps' \
                                             '/ComplexLogs/'
        elif self.logging_type == Logging.separated.name:
            self.__should_bootstrap_logging = False
            self.__should_separate_logging = True
            self._log_folder = log_folder or 'F:/PythonNotebooks/Study/Quantum/Bearings/experiments/Bootstraps' \
                                             '/SeparatedLogs/'

        self._should_logging = self.__should_bootstrap_logging or self.__should_separate_logging

    def run(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray],
            bearing_positive_ID: np.array = np.arange(100),
            bearing_negative_ID: np.array = np.arange(100, 112),
            verbose: bool = False):
        if verbose:
            print(f"logging {'enabled' if self._should_logging else 'disabled'}.")
            print(f"logging type: {'separate runs' if self.__should_bootstrap_logging else 'bootstrap'}.")

        if self._should_logging and not isinstance(self.splitter, Splitter):
            raise Exception("splits.Splitter() instance must be attached for logging. Use .set_splitter()")

        if self._should_logging and True:
            # TODO: Implement reducer validation
            pass
            # raise Exception("splits.Splitter() instance must be attached for logging")

        groups_count = len(bearing_negative_ID) + len(bearing_positive_ID)
        if X.shape[0] % groups_count != 0:
            raise ValueError(f'unable to split dataset into equal groups. Got {X.shape[0]} samples, and '
                             f'{len(bearing_negative_ID)} + {len(bearing_positive_ID)} groups')

        bootstrap_results = []
        x_processed = X.copy()

        if self.should_scale:
            x_processed = StandardScaler().fit_transform(x_processed)

        if isinstance(self.test_sample_size, int):
            train_ratio = 1 - self.test_sample_size / groups_count
        else:
            train_ratio = 1 - self.test_sample_size

        train_indices_log = []
        test_indices_log = []

        log_file_name = []

        for bootstrap_iteration in range(self.samples_number):
            train_bearing_id, test_bearing_id = BootstrapModeler._get_test_train_indices(bearing_negative_ID,
                                                                                         bearing_positive_ID,
                                                                                         train_ratio)
            x_train, x_test, y_train, y_test = self._split_test_train(x_processed, y,
                                                                      train_bearing_id, test_bearing_id)

            resampling_results = self.__get_bootstrap_scores(x_train, x_test, y_train, y_test)

            if self.__should_bootstrap_logging:
                train_indices_log.append(train_bearing_id.tolist())
                test_indices_log.append(test_bearing_id.tolist())

            if self.__should_separate_logging:
                for estimator_name in self.named_estimators.keys():
                    result_label = f"{self.__result_label_prefix}_bootstrap_{estimator_name}_{bootstrap_iteration}"
                    log_file_name.append(f"{result_label}.json")
                    current_estimator_results = resampling_results[estimator_name]
                    self._create_separate_log_file(
                        result_label=result_label,
                        result=current_estimator_results,
                        model_name=estimator_name,
                        resample_id=bootstrap_iteration,
                        train_indices=train_bearing_id.tolist(),
                        test_indices=test_bearing_id.tolist()
                    )

            bootstrap_results.append(resampling_results)

        if self.__should_bootstrap_logging:
            result_labels_list = [f"{self.__result_label_prefix}_bootstrap_{model_name}" for model_name
                                  in self.named_estimators.keys()]
            result_labels = dict(zip(self.named_estimators.keys(), result_labels_list))

            log_file_name = [f"{result_label}.json" for result_label in result_labels_list]
            self._create_experiment_log_file(bootstrap_results.copy(), result_labels,
                                             train_indices_log, test_indices_log)
        if not self._should_logging:
            return bootstrap_results.copy()
        else:
            return bootstrap_results.copy(), log_file_name

    def __get_bootstrap_scores(self, x_train, x_test, y_train, y_test):
        estimators_data = {}

        for estimator_name, estimator in zip(self.named_estimators.keys(), self.named_estimators.values()):
            estimator.fit(x_train, y_train)
            current_estimator_test_predictions = estimator.predict(x_test)

            current_estimator_results = {
                'accuracy': accuracy_score(y_test, current_estimator_test_predictions),
                'precision': precision_score(y_test, current_estimator_test_predictions),
                'recall': recall_score(y_test, current_estimator_test_predictions),
                'f1': f1_score(y_test, current_estimator_test_predictions),
                'predictions': current_estimator_test_predictions.tolist()
            }
            estimators_data[estimator_name] = current_estimator_results.copy()

        return estimators_data.copy()

    def _create_separate_log_file(self, result,  result_label, model_name, resample_id,
                                  train_indices, test_indices):
        results_logger = SingleRunResults(
            run_label=result_label,
            model_name=model_name,
            hyperparameters=self.named_estimators[model_name].get_params(),

            use_signal=self.splitter.use_signal,
            use_specter=self.splitter.use_specter,
            axes=self.splitter.frequency_data_columns,
            stats=list(self.splitter.stats.keys()),

            dim_reducing=self.should_reduce_dim,
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
            # TODO: Update reducing log info
        )
        write_result_obj_to_json(results_logger.dict(), f'{result_label}.json', self._log_folder)

    def _create_experiment_log_file(self, results, result_labels, train_ID, test_ID, ):
        models_names = self.named_estimators.keys()
        resampling_results_names = ['accuracy', 'precision', 'recall', 'f1', 'predictions']
        for model_name in models_names:
            result_label = result_labels[model_name]
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

                dim_reducing=self.should_reduce_dim,
                dim_reducing_method=self.reducing_method.name,
                # selected_features_ids = self.selected_features_ids
                selected_features_ids=[],

                train_brg_id=train_ID,
                test_brg_id=test_ID,
                predictions=model_scores['predictions'],

                accuracy_score=model_scores['accuracy'],
                precision_score=model_scores['precision'],
                recall_score=model_scores['recall'],
                f1_score=model_scores['f1'],

                resampling_number=self.samples_number
                # TODO: Update reducing log info
            )
            write_result_obj_to_json(results_logger.dict(), f"{result_label}.json", self._log_folder)

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
