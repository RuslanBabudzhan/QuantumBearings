"""
module implements work with csv and xlsx files for creating ML experiment tracking tables

Use generate_csv_from_log to convert *.json log files to new *.csv table

Use append_res_to_csv to append data from *.json log files to existing *.csv table

Use create_readable_xlsx to create human-readable *.xlsx table from *.csv table
"""

# TODO: rewrite converters based on standardized dataclass field objects

from typing import List, Optional
from collections import Counter
import re
import xlsxwriter

import pandas as pd
import numpy as np
from scipy import stats

from source.postprocessing.mljson import get_strings_from_jsons, get_result_obj_from_strings
from source.datamodels.datamodels import BaseResultsData, SingleRunResults, BootstrapResults, SingleDatasetsComparisonResults
from source.datamodels.datamodels import Axes, Stats


def convert_res_to_mapped_table_row(result_object: BaseResultsData) -> dict:
    """Converts single run result to dict, that can be writen as a experiment result table row"""
    converted = dict()

    for field_name, field in zip(result_object.__fields__.keys(), result_object.__fields__.values()):
        field_metadata = field.field_info.extra['metadata']
        if field_metadata['to_csv']:
            field_value = getattr(result_object, field_name)
            if field_metadata['enumerator']:
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


def generate_csv_from_log(results_names,
                          csv_name,
                          results_type,
                          results_path: Optional[str] = None,
                          csv_path: Optional[str] = None):
    if not bool(re.search("\.csv$", csv_name)):
        raise ValueError(f'csv file name must be in *.csv format. Got {csv_name}')
    if csv_path and not bool(re.search("/$", csv_path)):
        raise ValueError(f'path must end with "/" symbol.')
    results_strings = get_strings_from_jsons(results_names, results_path)
    results_objects = get_result_obj_from_strings(results_strings, results_type)
    results_dicts = []
    for res_obj in results_objects:
        results_dicts.append(convert_res_to_mapped_table_row(res_obj))
    joined_dict = full_dicts_join(results_dicts)

    results_df = pd.DataFrame(joined_dict)
    results_df.to_csv(f"{csv_path if csv_path else ''}{csv_name}", index=False)


def append_res_to_csv(results_names,
                      csv_name,
                      results_type,
                      results_path: Optional[str] = None,
                      csv_path: Optional[str] = None,
                      copy: bool = False):
    if not bool(re.search("\.csv$", csv_name)):
        raise ValueError(f'csv file name must be in *.csv format. Got {csv_name}')
    if csv_path and not bool(re.search("/$", csv_path)):
        raise ValueError(f'path must end with "/" symbol.')
    results_strings = get_strings_from_jsons(results_names, results_path)
    results_objects = get_result_obj_from_strings(results_strings, results_type)
    results_dicts = []
    for res_obj in results_objects:
        results_dicts.append(convert_res_to_mapped_table_row(res_obj))
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
