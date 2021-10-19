from typing import List, Optional
from collections import Counter
import re
import xlsxwriter

import pandas as pd
import numpy as np
from scipy import stats

from source.postprocessing.mljson import get_strings_from_jsons, get_result_obj_from_strings
from source.datamodels.datamodels import SingleRunResults, BootstrapResults, Axes, Stats


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


def convert_bootstrap_res_to_dict(result_object: BootstrapResults, hyperparams_need: bool = False) -> dict:
    """Converts single run result to dict, that can be writen as a experiment result table row"""
    simple_fields = {
        "run label": result_object.run_label,
        "ML model": result_object.model_name,
        "Signals data used": result_object.use_signal,
        "Specter data used": result_object.use_specter,
        "samplings number": result_object.resampling_number,
        "F1 mean": np.mean(result_object.f1_score),
        "F1 median": np.median(result_object.f1_score),
        "F1 mode": stats.mode(np.array(result_object.f1_score))[0][0],
        "F1 std": np.std(result_object.f1_score),
        "F1 skewness": stats.skew(np.array(result_object.f1_score)),
        "F1 Kurtosis": stats.kurtosis(np.array(result_object.f1_score))
    }

    converted = dict()
    for field_name, field in zip(simple_fields.keys(), simple_fields.values()):
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
    results_strings = get_strings_from_jsons(results_names, results_path)
    results_objects = get_result_obj_from_strings(results_strings, SingleRunResults)
    results_dicts = []
    for res_obj in results_objects:
        results_dicts.append(convert_single_res_to_dict(res_obj, hyperparams_need=False))
    joined_dict = full_dicts_join(results_dicts)

    results_df = pd.DataFrame(joined_dict)
    results_df.to_csv(f"{csv_path if csv_path else ''}{csv_name}", index=False)


def generate_csv_from_bootstrap_results_log(results_names,
                                            csv_name,
                                            results_path: Optional[str] = None,
                                            csv_path: Optional[str] = None):
    if not bool(re.search("\.csv$", csv_name)):
        raise ValueError(f'csv file name must be in *.csv format. Got {csv_name}')
    if csv_path and not bool(re.search("/$", csv_path)):
        raise ValueError(f'path must end with "/" symbol.')
    results_strings = get_strings_from_jsons(results_names, results_path)
    results_objects = get_result_obj_from_strings(results_strings, BootstrapResults)
    results_dicts = []
    for res_obj in results_objects:
        results_dicts.append(convert_bootstrap_res_to_dict(res_obj, hyperparams_need=False))
    joined_dict = full_dicts_join(results_dicts)

    results_df = pd.DataFrame(joined_dict)
    results_df.to_csv(f"{csv_path if csv_path else ''}{csv_name}", index=False)


def append_single_res_to_csv(results_names,
                             csv_name,
                             results_path: Optional[str] = None,
                             csv_path: Optional[str] = None,
                             copy: bool = False):
    if not bool(re.search("\.csv$", csv_name)):
        raise ValueError(f'csv file name must be in *.csv format. Got {csv_name}')
    if csv_path and not bool(re.search("/$", csv_path)):
        raise ValueError(f'path must end with "/" symbol.')
    results_strings = get_strings_from_jsons(results_names, results_path)
    results_objects = get_result_obj_from_strings(results_strings, SingleRunResults)
    results_dicts = []
    for res_obj in results_objects:
        results_dicts.append(convert_single_res_to_dict(res_obj, hyperparams_need=False))
    joined_dict = full_dicts_join(results_dicts)

    old_results_df = pd.read_csv(f"{csv_path if csv_path else ''}{csv_name}")
    new_results_df = pd.DataFrame(joined_dict)
    results_df = pd.concat([old_results_df, new_results_df], ignore_index=True)
    if copy:
        results_df.to_csv(f"{csv_path if csv_path else ''}copy_{csv_name}", index=False)
    else:
        results_df.to_csv(f"{csv_path if csv_path else ''}{csv_name}", index=False)


def append_bootstrap_res_to_csv(results_names,
                                csv_name,
                                results_path: Optional[str] = None,
                                csv_path: Optional[str] = None,
                                copy: bool = False):
    if not bool(re.search("\.csv$", csv_name)):
        raise ValueError(f'csv file name must be in *.csv format. Got {csv_name}')
    if csv_path and not bool(re.search("/$", csv_path)):
        raise ValueError(f'path must end with "/" symbol.')
    results_strings = get_strings_from_jsons(results_names, results_path)
    results_objects = get_result_obj_from_strings(results_strings, BootstrapResults)
    results_dicts = []
    for res_obj in results_objects:
        results_dicts.append(convert_bootstrap_res_to_dict(res_obj, hyperparams_need=False))
    joined_dict = full_dicts_join(results_dicts)

    old_results_df = pd.read_csv(f"{csv_path if csv_path else ''}{csv_name}")
    new_results_df = pd.DataFrame(joined_dict)
    results_df = pd.concat([old_results_df, new_results_df], ignore_index=True)
    if copy:
        results_df.to_csv(f"{csv_path if csv_path else ''}copy_{csv_name}", index=False)
    else:
        results_df.to_csv(f"{csv_path if csv_path else ''}{csv_name}", index=False)


def create_readable_xlsx(xlsx_name, csv_name, xlsx_path: Optional[str] = None, csv_path: Optional[str] = None):
    themes = {'light blue': {'head_font': '#F3F3F3', 'head_bg': '#142459', 'font': '#3D3D3D', 'bg': '#FBFBFB',
                             'head_border': '#F3F3F3', 'border': '#6A6A6A'}}
    theme = themes['light blue']
    head_format_dict = {'bold': True,
                        'font_size': 14,
                        'valign': "down",
                        'font_color': theme['head_font'],
                        'bg_color': theme['head_bg'],
                        'border': 2}

    line_format_dict = {'font_size': 14,
                        "align": "right",
                        'font_color': theme['font'],
                        'bg_color': theme['bg'],
                        'border': 1}

    if not bool(re.search("\.csv$", csv_name)):
        raise ValueError(f'csv file name must be in *.csv format. Got {csv_name}')
    if not bool(re.search("\.xlsx$", xlsx_name)):
        raise ValueError(f'xlsx file name must be in *.xlsx format. Got {xlsx_name}')
    if xlsx_path and not bool(re.search("/$", xlsx_path)):
        raise ValueError(f'xlsx path must end with "/" symbol. Got {xlsx_path}')
    if csv_path and not bool(re.search("/$", csv_path)):
        raise ValueError(f'csv path must end with "/" symbol. Got {csv_path}')

    data = pd.read_csv(f"{csv_path if csv_path else ''}{csv_name}", delimiter=',')
    data.insert(loc=0, column='experiment index', value=np.arange(len(data)))
    data.astype('object')

    workbook = xlsxwriter.Workbook(f"{xlsx_path if xlsx_path else ''}{xlsx_name}")
    worksheet = workbook.add_worksheet()

    bool_map = {True: "Yes", False: "No", None: None}
    float_map = lambda x: str(np.round(x, 3)).replace('.', ',') \
        if isinstance(x, (float, np.float, np.float64)) and not np.isnan(x) else ""

    for col_id, column in enumerate(data):
        column_format = workbook.add_format(line_format_dict)
        column_format.set_border_color(theme['border'])
        if any(data[column].map(lambda x: isinstance(x, (float, np.float, np.float64)) and not np.isnan(x))):
            prepared_column = data[column].map(float_map).fillna("")
        elif any(data[column].map(lambda x: isinstance(x, (bool, np.bool, np.bool_)))):
            prepared_column = data[column].map(bool_map).fillna("")
        else:
            prepared_column = data[column].fillna("")

        col_width = 1.5 * max([len(column), max(prepared_column.map(lambda x: len(str(x))))])
        worksheet.set_column(col_id, col_id, col_width, column_format)
        worksheet.write_column(1, col_id, prepared_column)

    head_format = workbook.add_format(head_format_dict)
    head_format.set_border_color(theme['head_border'])
    worksheet.set_row(0, 22, head_format)
    worksheet.write_row(0, 0, data.columns)

    workbook.close()
