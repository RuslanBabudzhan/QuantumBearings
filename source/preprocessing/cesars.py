from typing import Optional, List, Union
import re
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
from scipy import io


class CesarPreprocessor:
    @staticmethod
    def parse_directory(directory: str, data_keys: Union[List[str], str] = "Rod_1",
                        filenames: Optional[List[str]] = None):
        """
        creates dataframe from *.mat files.
        :param directory:
        :param data_keys:
        :param filenames:
        :return: signals dataframe, targets dataframe
        """
        sampling_time = 30
        if not bool(re.search("/$", directory)):
            raise ValueError(f'dir must end with "/" symbol.')
        if not filenames:
            pass
            filenames = [f for f in listdir(directory) if isfile(join(directory, f))]
        for filename in filenames:
            if not bool(re.search("\.mat$", filename)):
                raise ValueError(f'mat file name must be in *.mat format. Got {filename}')

        if isinstance(data_keys, str):
            data_keys = [data_keys]

        signals_data_dict = {'experiment_id': [], 'timestamp': []}
        for col in data_keys:
            signals_data_dict[col] = []
        targets_data_dict = {'bearing_id': [], 'status': []}
        for file_id, filename in enumerate(filenames):
            targets_data_dict['bearing_id'].append(file_id)
            targets_data_dict['status'].append(1 if bool(re.search("_F0_", filename)) else 0)

            datafile = io.loadmat(f"{directory}{filename}")
            records_count = len(datafile[data_keys[0]])
            timestamp = np.linspace(0, sampling_time, records_count)
            signals_data_dict['experiment_id'].extend(list(np.repeat(file_id, records_count, axis=0)))
            signals_data_dict['timestamp'].extend(list(timestamp))

            for signal_column in data_keys:
                signals_data_dict[signal_column].extend(list(datafile[signal_column]))

        return pd.DataFrame(signals_data_dict), pd.DataFrame(targets_data_dict)
