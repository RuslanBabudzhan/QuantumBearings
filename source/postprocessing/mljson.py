import json
from typing import Union, List, Type
import dataclasses

import pydantic

from datamodels import BaseResultsData


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def write_result_obj_to_json(result: dict, filename: str, filepath: str = None):
    if filepath:
        fullname = f"{filepath}/{filename}.json"
    else:
        fullname = f"{filename}.json"
    with open(fullname, "w") as write_file:
        json.dump(result, write_file, cls=EnhancedJSONEncoder)


def get_strings_from_jsons(filenames: Union[str, List[str]], filepath: str = None) -> List[str]:
    if isinstance(filenames, str):
        filenames = [filenames]
    json_strings = []
    for filename in filenames:
        if filepath:
            fullname = f"{filepath}/{filename}.json"
        else:
            fullname = f"{filename}.json"
        with open(fullname, "r") as read_file:
            json_strings.append(read_file.read())
    return json_strings.copy()


def get_result_obj_from_strings(json_strings: Union[str, List[str]],
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
