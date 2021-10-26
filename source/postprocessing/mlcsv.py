"""
module implements work with *.csv and *.xlsx files for creating ML experiment tracking tables

Use generate_csv_from_result to convert results objects/files to new *.csv table

Use append_results_to_csv to append data from *.json log files to existing *.csv table

Use create_readable_xlsx to create human-readable *.xlsx table from *.csv table
"""

from typing import List, Union, Optional, Type
from collections import Counter
import re
import xlsxwriter

import pandas as pd
import numpy as np

from source.postprocessing.mljson import deserialize_result
from source.datamodels.datamodels import BaseResultsData


def convert_res_to_mapped_table_row(result_object: BaseResultsData) -> dict:
    """Converts single run result to dict, that can be writen as a experiment result table row"""
    converted = dict()

    for field_name, field in zip(result_object.__fields__.keys(), result_object.__fields__.values()):
        field_metadata = field.field_info.extra['metadata']
        if field_metadata['to_csv']:
            field_value = getattr(result_object, field_name)
            if field_metadata['enumerator'] and isinstance(field_value, list):
                field_encoder = field_metadata['enumerator']
                for possible_value in field_encoder.get_keys():
                    converted[f"{field_metadata['short_description']}: {possible_value}"] = \
                        True if possible_value in field_value else False
            elif isinstance(field_value, dict):
                for parameter_name, parameter_value in zip(field_value.keys(), field_value.values()):
                    converted[f"{field_metadata['short_description']}: {parameter_name}"] = parameter_value
            else:
                converted[field_name] = field_value
    return converted


def _full_dicts_join(dicts: List[dict]) -> dict:
    """
    Full joining of dictionaries. Used for stack rows of result tables
    Produces dict with keys that are the union of the key sets of the input dictionaries and
    values that are lists of input dictionaries values. Empty values are filled with None

    Example:
        >> d1 = {'a': 4, 'b': 12.5, 'c': 'ans', 'd': [5.1, 16]}
        >> d2 = {'b': 5.4, 'a': 11, 'c': 'res', 'e': True}
        >> d = _full_dicts_join([d1, d2])
        >> print(d)
        {'a': [4, 11], 'b': [12.5, 5.4], 'c': ['ans', 'res'], 'd': [[5.1, 16], None], 'e': [None, True]}
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


def generate_csv_from_results(results: Union[List[BaseResultsData], List[str]],
                              csv_name: str,
                              results_type: Type[BaseResultsData],
                              results_path: Optional[str] = None,
                              csv_path: Optional[str] = None):
    """
    Creates new *.csv file, containing rows, that represent ML experiments results
    :param results: result objects or names of files that represent rows to fill in the *.csv file
    :param csv_name: name of *.csv file
    :param results_type: type of results. Use source.datamodels.datamodels
    :param results_path: path to reach files with results data
    :param csv_path: path to save *.csv file
    :return: None
    """
    if not bool(re.search("\.csv$", csv_name)):
        raise ValueError(f'csv file name must be in *.csv format. Got {csv_name}')
    if csv_path and not bool(re.search("/$", csv_path)):
        raise ValueError(f'path must end with "/" symbol.')
    if isinstance(results[0], str):
        results_objects = deserialize_result(results, results_type, results_path)
    else:
        results_objects = results
    results_dicts = []
    for res_obj in results_objects:
        results_dicts.append(convert_res_to_mapped_table_row(res_obj))
    joined_dict = _full_dicts_join(results_dicts)

    results_df = pd.DataFrame(joined_dict)
    results_df.to_csv(f"{csv_path if csv_path else ''}{csv_name}", index=False)


def append_results_to_csv(results: Union[List[BaseResultsData], List[str]],
                          csv_name: str,
                          results_type: Optional[Type[BaseResultsData]],
                          results_path: Optional[str] = None,
                          csv_path: Optional[str] = None,
                          copy: bool = False):
    """
    Append result rows to existing *.csv file. Each row represents ML experiments results
    :param results: result objects or names of files that represent rows to fill in the *.csv file
    :param csv_name: name of *.csv file
    :param results_type: type of results. Use source.datamodels.datamodels
    :param results_path: path to reach files with results data
    :param csv_path: path to save *.csv file
    :param copy: Append results to the copy of existing *.csv or rewrite the *.csv
    :return: None
    """
    if not bool(re.search("\.csv$", csv_name)):
        raise ValueError(f'csv file name must be in *.csv format. Got {csv_name}')
    if csv_path and not bool(re.search("/$", csv_path)):
        raise ValueError(f'path must end with "/" symbol.')
    if isinstance(results[0], str):
        if not results_type:
            raise TypeError('"results_type" must be added if "results" represents a list of files names')
        results_strings = _get_strings_from_jsons(results, results_path)
        results_objects = _get_result_obj_from_strings(results_strings, results_type)
    else:
        results_objects = results
    results_dicts = []
    for res_obj in results_objects:
        results_dicts.append(convert_res_to_mapped_table_row(res_obj))
    joined_dict = _full_dicts_join(results_dicts)

    old_results_df = pd.read_csv(f"{csv_path if csv_path else ''}{csv_name}")
    new_results_df = pd.DataFrame(joined_dict)
    results_df = pd.concat([old_results_df, new_results_df], ignore_index=True)
    if copy:
        results_df.to_csv(f"{csv_path if csv_path else ''}copy_{csv_name}", index=False)
    else:
        results_df.to_csv(f"{csv_path if csv_path else ''}{csv_name}", index=False)


def create_readable_xlsx(xlsx_name, csv_name, xlsx_path: Optional[str] = None, csv_path: Optional[str] = None):
    """
    Creates readable xlsx file from csv name.
    :param xlsx_name: name of *.xlsx file
    :param csv_name: name of *.csv file
    :param xlsx_path: path of *.xlsx file
    :param csv_path: path of *.csv file
    :return: None
    """
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
    dataframe_columns = list(data.columns)
    sorted_columns_order = ['Scores', 'Axes', 'Statistics', 'Hyperparameters']
    for columns_group_prefix in sorted_columns_order:
        group_pattern = f"^{columns_group_prefix}:"
        group_matches = [col_name for col_name in dataframe_columns if re.search(group_pattern, col_name)]
        dataframe_columns = [col_name for col_name in dataframe_columns if col_name not in group_matches]
        dataframe_columns.extend(group_matches)
    data = data[dataframe_columns]

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
