from typing import List
from collections import Counter

from datamodels import SingleRunResults, Axes, Stats


def convert_single_res_to_dict(result_object: SingleRunResults, hyperparams_need: bool = False) -> dict:
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
