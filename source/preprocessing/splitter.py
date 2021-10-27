from typing import Tuple, List, Optional

import numpy as np
import pandas as pd

from source.preprocessing.basesplitter import BaseSplitter
# TODO: ValueError for stable_area as a list of tuples

class Splitter(BaseSplitter):
    DATA_REQUIRED_COLUMNS = ['target', 'experiment_id', 'timestamp']

    def split_dataset(self,
                      dataset: pd.DataFrame,
                      stable_area: Optional[List[Tuple[int, int]]] = None,
                      splits_number: int = 10,
                      signal_data_columns: List[str] = None) -> pd.DataFrame:
        """
        Split dataset by chunks and return dataset with statistics of the chunks
        :param dataset: dataset with signals data
        :param stable_area: list of time intervals in which the signal is stable and must be processed.
            The intervals must fall within the values of 'timestamp' dataset column
        :param splits_number: into how many sections it needs to divide the signal of one experiment
        :param signal_data_columns: names of dataset columns, that contain signal data
        :return: dataframe with statistics of signals
        """
        self.splits_number = splits_number
        if stable_area is None:
            stable_area = [(10, 20)]
        self.stable_area = stable_area
        if signal_data_columns is None and self.signal_data_columns is None:
            self.signal_data_columns = ['a1_x', 'a1_y', 'a1_z', 'a2_x', 'a2_y', 'a2_z']
        if signal_data_columns is not None:
            self.signal_data_columns = signal_data_columns

        required_data_columns = Splitter.DATA_REQUIRED_COLUMNS.copy()
        required_data_columns.extend(self.signal_data_columns)
        for column in required_data_columns:
            if column not in dataset.columns:
                raise ValueError(f"dataset must have {column} column")

        experiments_indices = dataset['experiment_id'].unique()

        sliced_dataset = dataset.copy()
        if stable_area:
            stable_zone_mask = sliced_dataset['timestamp'].between(stable_area[0][0], stable_area[0][1])
            for timezone in stable_area[1:]:
                stable_zone_mask = stable_zone_mask | sliced_dataset['timestamp'].between(timezone[0], timezone[1])
            sliced_dataset = sliced_dataset[stable_zone_mask]
            sliced_dataset.reset_index(drop=True, inplace=True)

        one_experiment_records_count = len(sliced_dataset[sliced_dataset['experiment_id'] == experiments_indices[0]])
        records_to_drop_count = one_experiment_records_count % self.splits_number

        for experiment in range(len(experiments_indices)):
            stop = (one_experiment_records_count - records_to_drop_count)*experiment + one_experiment_records_count
            start = stop - records_to_drop_count
            drop_zone = np.arange(start+1, stop+1)
            sliced_dataset.drop(labels=list(drop_zone), axis=0, inplace=True)

        features_matrices = []
        resulting_rows_count = self.splits_number * len(experiments_indices)
        for signal_name in self.signal_data_columns:
            signal_vector = sliced_dataset[signal_name].to_numpy()
            signals_data = signal_vector.reshape(resulting_rows_count, -1)
            stat_data = self._get_data_statistics(signals_data)
            features_matrices.append(stat_data)

        prepared_dataset = np.hstack(features_matrices)
        targets = dataset[['experiment_id', 'target']].groupby('experiment_id', as_index=False).max()
        needed_targets = targets[targets['experiment_id'].isin(experiments_indices)]
        targets_vector = needed_targets['target'].to_numpy().repeat(self.splits_number).reshape(-1, 1)
        groups_vector = experiments_indices.repeat(self.splits_number).reshape(-1, 1)
        prepared_dataset = np.hstack([targets_vector, groups_vector, prepared_dataset])

        dataset_columns = ['target', 'group']
        signals_stats_columns = []
        stats_columns_suffixes = self.stats.keys()
        data_types_suffixes = []
        if self.use_signal:
            data_types_suffixes.append('signal')
        if self.use_specter:
            data_types_suffixes.append('specter')

        extended_data_columns = []
        for signal_name in self.signal_data_columns:
            extended_data_columns.extend([f"{signal_name}_{suffix}" for suffix in data_types_suffixes])

        for signal_name in extended_data_columns:
            one_signal_stats_names = [f"{signal_name}_{suffix}" for suffix in stats_columns_suffixes]
            signals_stats_columns.extend(one_signal_stats_names)

        dataset_columns.extend(signals_stats_columns)
        prepared_dataset = pd.DataFrame(prepared_dataset, columns=dataset_columns)
        return prepared_dataset
