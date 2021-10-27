"""

Module implements serialization and deserialization of Results objects (from source.datamodels.datamodels)

Use serialize_result to serialize Result object

Use deserialize_result to deserialize Result objects

"""

import json
import re
from typing import Union, Optional, List, Type
import dataclasses

import pydantic

from source.datamodels.datamodels import BaseResultsData


class EnhancedJSONEncoder(json.JSONEncoder):
    """Used for serialization of pydantic`s dataclasses objects"""

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def serialize_results(results: Union[BaseResultsData, List[BaseResultsData]],
                      filenames: Union[str, List[str]], filepath: Optional[str] = None):
    """
    implements Results object serialization
    :param results: pydantic objects results of ML experiment
    :param filenames: names of *.json files
    :param filepath: path to *.json files
    :return: None
    """

    if isinstance(filenames, str):
        filenames = [filenames]
    if isinstance(results, BaseResultsData):
        results = [results]
    for filename in filenames:
        if not bool(re.search("\.json$", filename)):
            raise ValueError(f'file name must be in *.json format. Got {filename}')
    if filepath and not bool(re.search("/$", filepath)):
        raise ValueError(f'path must end with "/" symbol.')

    for result, filename in zip(results, filenames):
        result = result.dict()
        if filepath:
            fullname = f"{filepath}{filename}"
        else:
            fullname = f"{filename}"
        with open(fullname, "w") as write_file:
            json.dump(result, write_file, cls=EnhancedJSONEncoder)


def deserialize_results(filenames: Union[str, List[str]],
                        result_obj_type: Type[BaseResultsData],
                        filepath: Optional[str] = None) -> List[BaseResultsData]:
    """
    implements Results object deserialization
    :param filenames: names of *.json files that need to be deserialized
    :param result_obj_type: type of objects to which files should be casted
    :param filepath: path to *.json files
    :return: list of objects of type result_obj_type
    """
    json_strings = _get_strings_from_jsons(filenames, filepath)
    result_objects = _get_result_obj_from_strings(json_strings, result_obj_type)
    return result_objects


def _get_strings_from_jsons(filenames: Union[str, List[str]], filepath: Optional[str] = None) -> List[str]:
    """generates strings from *.json files for parsing"""
    if isinstance(filenames, str):
        filenames = [filenames]

    if not bool(re.search("\.json$", filenames[0])):
        raise ValueError(f'log file name must be in *.json format. Got {filenames[0]}')
    if filepath and not bool(re.search("/$", filepath)):
        raise ValueError(f'path must end with "/" symbol.')

    json_strings = []
    for filename in filenames:
        if filepath:
            fullname = f"{filepath}{filename}"
        else:
            fullname = f"{filename}"
        with open(fullname, "r") as read_file:
            json_strings.append(read_file.read())
    return json_strings.copy()


def _get_result_obj_from_strings(json_strings: Union[str, List[str]],
                                 result_obj_type: Type[BaseResultsData]) -> List[BaseResultsData]:
    """ implements parsing of json-formatted string to specified ResultData object """
    if isinstance(json_strings, str):
        json_strings = [json_strings]
    result_objects = []
    for string_index, string in enumerate(json_strings):
        try:
            result_objects.append(result_obj_type.parse_raw(string))
        except pydantic.error_wrappers.ValidationError:
            raise Exception(f"JSON string under index #{string_index} does not match {result_obj_type}")
    return result_objects.copy()
