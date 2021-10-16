from typing import List, Optional
from collections import Counter
import re

import pandas as pd

from source.postprocessing.mljson import get_strings_from_jsons, get_result_obj_from_strings
from source.datamodels.datamodels import SingleRunResults, Axes, Stats


def convert_single_res_to_dict(result_object: SingleRunResults, hyperparams_need: bool = False) -> dict:
    """Converts single run result to dict, that can be writen as a experiment result table row"""
    single_run_simple_fields = {
        "run label": result_object.run_label,
        "ML model": result_object.model_name,
        "Signals data used": result_object.use_signal,
        "Specter data used": result_object.use_specter,
        "Accuracy": result_object.accuracy_score,
        "Precision": result_object.precision_score,
        "Recall": result_object.recall_score,
        "F1": result_object.f1_score
    }

    converted = dict()
    for field_name, field in zip(single_run_simple_fields.keys(), single_run_simple_fields.values()):
        converted[field_name] = field

    for axis in Axes.get_keys():
        converted["Axis: " + axis] = True if axis in result_object.axes else False

    for stat in Stats.get_keys():
        converted["Statistic: " + stat] = True if stat in result_object.stats else False

    if hyperparams_need:
        for param_name, param in zip(result_object.hyperparameters.keys(), result_object.hyperparameters.values()):
            converted["HyperParameter: " + param_name] = param

    return converted


def full_dicts_join(dicts: List[dict]) -> dict:
    """
    Full joining of dictionaries.
    Produces dict with keys that are the union of the key sets of the input dictionaries and
    values that are lists of input dictionaries values. Empty values are filled with None
    """
    joined_dict_keys = [list(d.keys()) for d in dicts]
    joined_dict_keys = Counter([item for sublist in joined_dict_keys for item in sublist])
    joined = dict([(key, []) for key in joined_dict_keys])
    for d in dicts:
        d_keys = list(d.keys())
        for key in joined_dict_keys:
            if key in d_keys:
                joined[key].append(d[key])
            else:
                joined[key].append(None)

    return joined


def generate_csv_from_single_results_log(results_names,
                                         csv_name,
                                         results_path: Optional[str] = None,
                                         csv_path: Optional[str] = None):
    if not bool(re.search("\.csv$", csv_name)):
        raise ValueError(f'csv file name must be in *.csv format. Got {csv_name}')
    if csv_path and not bool(re.search("/$", csv_path)):
        raise ValueError(f'path must end with "/" symbol.')
    print(results_path)
    results_strings = get_strings_from_jsons(results_names, results_path)
    results_objects = get_result_obj_from_strings(results_strings, SingleRunResults)
    results_dicts = []
    for res_obj in results_objects:
        results_dicts.append(convert_single_res_to_dict(res_obj, hyperparams_need=False))
    joined_dict = full_dicts_join(results_dicts)

    results_df = pd.DataFrame(joined_dict)
    if csv_path:
        results_df.to_csv(f"{csv_path}{csv_name}")
    else:
        results_df.to_csv(csv_name)

    pass
