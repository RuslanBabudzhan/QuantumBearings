"""

Module implements serialization and deserialization of Results objects (from source.datamodels.datamodels)

Use serialize_result to serialize Result object

Use deserialize_result to deserialize Result objects

"""
import os
import configparser
import json
import re
from typing import Union, Optional, List, Type

import pydantic
import dataclasses
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from tqdm import tqdm

from source.datamodels.datamodels import BaseResultsData
from source.utils import get_project_root


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

    for result, filename in zip(results, filenames):
        result = result.dict()
        if filepath:
            fullname = os.path.join(filepath, filename)
        else:
            fullname = filename
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

    json_strings = []
    for filename in filenames:
        if filepath:
            fullname = os.path.join(filepath, filename)
        else:
            fullname = filename
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


def upload_to_Drive(drive_folder_key: str, filepath: Optional[str] = None, filenames: Optional[List[str]] = None):
    """
    Uploads files to Google Drive
    :param filenames: list of strings, optional. Names of files need to be uploaded. If None, all files from filepath
        will be uploaded to the drive
    :param filepath: string, optional. Path to search for files in filenames. If None, Root folder will be used
    :param drive_folder_key: string. Key to specify folder on Google Drive
    :return: None
    """
    root = get_project_root()
    userconfig = configparser.ConfigParser()
    userconfig.read(os.path.join(root, "userconfig.ini"))

    if not filepath:
        print(f"filepath is not given. Search in {root}")
        filepath = root

    if not os.path.isdir(filepath):
        filepath = os.path.join(root, filepath)
        if not os.path.isdir(filepath):
            raise ValueError(f"The system cannot find the path {filepath}.")

    config = configparser.ConfigParser()
    config.read(os.path.join(root, "config.ini"))
    folder_ids = dict()
    for field in config['Drive']:
        folder_ids[field] = config['Drive'][field]
    if drive_folder_key not in set(folder_ids.keys()):
        raise ValueError(f"Unknown Drive folder key. Got: '{drive_folder_key}'. Expected: {set(folder_ids.keys())}")
    if filenames is None:
        filenames = set((file for file in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, file))))
        print(f"filenames are not given. All files will be uploaded ({len(filenames)})")
    folder_id = folder_ids[drive_folder_key]

    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("mycreds.txt")
    if gauth.credentials is None:
        gauth.LocalWebserverAuth()  # Authenticate if they're not there
    elif gauth.access_token_expired:
        gauth.Refresh()  # Refresh them if expired
    else:
        gauth.Authorize()  # Initialize the saved creds
    gauth.SaveCredentialsFile("mycreds.txt")  # Save the current credentials to a file

    drive = GoogleDrive(gauth)

    file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(folder_id)}).GetList()
    existing_filenames = [file['title'] for file in file_list]

    for filename in tqdm(filenames):
        upload_file = _create_non_duplicating_name(filename, existing_filenames)
        if upload_file is not filename:
            print(f"File {filename} already exists in Drive directory. New file saved as {upload_file}")
        gfile = drive.CreateFile({'title': upload_file, 'parents': [{'id': folder_id}]})
        gfile.SetContentFile(os.path.join(filepath, filename))  # os.path.join(filepath, upload_file)
        gfile.Upload(param={'supportsTeamDrives': True})  # Upload the file.


def _create_non_duplicating_name(filename, existing_names, prefix='copy_'):
    if filename in existing_names:
        filename = f"{prefix}{filename}"
        return _create_non_duplicating_name(filename, existing_names, prefix)
    else:
        return filename
