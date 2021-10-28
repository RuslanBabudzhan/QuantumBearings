"""

This module implements models of data, used for ML experiments results tracking

"""

# TODO: Add validation for fields with enumerators
# TODO: Add validation for length equality of arrays
# TODO: Change specter_threshold description

from typing import List, Dict, Optional
from abc import ABC

from pydantic import BaseModel, Field

from source.datamodels.iterators import Axes, Stats, Metrics


class BaseResultsData(ABC, BaseModel):
    """ Abstract model of data obtained as a result of ML algorithm launch """
    run_label: str = Field(metadata=dict(short_description="Run label", to_csv=True, printable=True, enumerator=None,
                                         long_description="Label assigned to the result"))
    model_name: str = Field(metadata=dict(short_description="ML model", to_csv=True, printable=True, enumerator=None,
                                          long_description="Name of the model (GBM, RF, etc.)"))
    hyperparameters: dict = Field(metadata=dict(short_description="Hyperparameters", to_csv=True,
                                                printable=False, enumerator=None, long_description="dict with "
                                                "hyperparameters names as keys and hyperparameters values as values"))

    use_signal: bool = Field(metadata=dict(short_description="Was signal used", to_csv=True, printable=True,
                                           enumerator=None, long_description="Was raw signal data used in training"))
    use_specter: bool = Field(metadata=dict(short_description="Was specter used", to_csv=True, printable=True,
                                            enumerator=None, long_description="Was specter of signal used in training"))
    specter_threshold: Optional[int] = Field(default_factory=None, metadata=dict(short_description="Specter threshold",
                                             printable=True, enumerator=None, to_csv=True,
                                             long_description="Max frequency in signal specter"))

    axes: List[str] = Field(metadata=dict(short_description="Axes", to_csv=True, printable=False,
                                          enumerator=Axes, long_description="Which axes were used in training. "
                                                                            "Use Axes.<axis>.name"))
    stats: List[str] = Field(metadata=dict(short_description="Statistics", to_csv=True, printable=False,
                                           enumerator=Stats, long_description="Which statistics were used in training. "
                                                                              "Use Stats.<stat>.name"))

    predictions: List[float] = Field(metadata=dict(short_description="Model predictions", to_csv=False, printable=False,
                                                   enumerator=None, long_description="Prediction for each test sample"))
    scores: Dict[str, float] = Field(metadata=dict(short_description="Scores", to_csv=True, printable=True,
                                                   enumerator=Metrics, long_description=" Dict of scores (direct/mean),"
                                                   " for keys use Metrics.<metric>.name"))

    def __str__(self):
        result_string = ""
        for field_name, field in zip(self.__fields__.keys(), self.__fields__.values()):
            field_metadata = field.field_info.extra['metadata']
            if field_metadata['printable']:
                value = getattr(self, field_name)
                result_string += f"{field_metadata['short_description']}: {str(value)}\n"
        return result_string


class SingleRunResults(BaseResultsData):
    """ Model of data obtained as a result of single ML algorithm launch """
    train_brg_id: List[int] = Field(metadata=dict(short_description="Train ID", to_csv=False, printable=False,
                                                  enumerator=None, long_description="Bearings IDs in train subsample"))
    test_brg_id: List[int] = Field(metadata=dict(short_description="Test ID", to_csv=False, printable=False,
                                                 enumerator=None, long_description="Bearings IDs in test subsample"))


class SingleDatasetsComparisonResults(SingleRunResults):
    """ Model of data obtained as a result of datasets comparison for one fitting of ML model."""
    train_dataset_name: str = Field(metadata=dict(short_description="Train DF", to_csv=True, printable=True,
                                                  enumerator=None, long_description="Dataset used for train"))
    test_dataset_name: str = Field(metadata=dict(short_description="Test DF", to_csv=True, printable=True,
                                                 enumerator=None, long_description="Dataset used for test"))
    signal_scaler: str = Field(metadata=dict(short_description="Scaler", to_csv=True, printable=True, enumerator=None,
                                             long_description=" Scaler of raw data. Use Scalers.<scaler>.name"))


class BootstrapResults(BaseResultsData):
    """ Model of data obtained as a result of Bootstrap over single ML algorithm """
    resampling_number: int = Field(metadata=dict(short_description="Resamples", to_csv=True, printable=True,
                                                 enumerator=None, long_description=" number of model fits"))

    predictions: Optional[List[List[float]]] = Field(default_factory=None,
                                                     metadata=dict(short_description="Predictions", to_csv=False,
                                                     printable=False, enumerator=None, long_description="Prediction"
                                                     " for each test sample in each resampling"))
    bootstrap_scores: Dict[str, List[float]] = Field(metadata=dict(short_description="Scores", to_csv=False,
                                                                   printable=False, enumerator=Metrics,
                                                                   long_description="Dict of scores for each "
                                                                   "resampling, for keys use Metrics.<metric>.name"))


class BootstrapFeatureSelectionResults(BootstrapResults):
    """ Bootstrap to perform feature selection experiment """
    selector_name: str = Field(metadata=dict(short_description="Selector", to_csv=True, printable=True, enumerator=None,
                                             long_description="Name of features selector method"))
    selector_model_name: Optional[str] = Field(default_factory="", metadata=dict(short_description="Selector model",
                                                                                 printable=True, enumerator=None,
                                                                                 to_csv=True, long_description="Name of"
                                                                                 " ML model (if required)"))

    selector_model_hyperparameters: dict = Field(default_factory={}, metadata=dict(short_description="Hyperparameters "
                                                 "of ML model", to_csv=True, printable=False, enumerator=None,
                                                 long_description="dict with selector hyperparameters names as keys and"
                                                 " hyperparameters values as values"))

    ranked_features_id: List[int] = Field(metadata=dict(short_description="Ranked features", to_csv=False,
                                                        printable=False, enumerator=None, long_description="List of "
                                                        "features sorted by the splitter in descending order of "
                                                        "importance"))

    threshold: int = Field(metadata=dict(short_description="Features rank threshold", to_csv=False, printable=False,
                                         enumerator=None, long_description=" id of last important feature"))


class GridSearchResults(BootstrapResults):
    """ Model of data obtained as a result of model tuning """
    hyperparameters_grid: dict = Field(metadata=dict(short_description="Search grid", to_csv=False, printable=False,
                                                     enumerator=None, long_description="dict with hyperparameters names"
                                                     " as keys and lists of hyperparameters values as values"))


class BootstrapDatasetsComparisonResults(SingleDatasetsComparisonResults, BootstrapResults):
    """ Model of data obtained as a result of datasets comparison for fitting of ML model with bootstrap resampling."""
