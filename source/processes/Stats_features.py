import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, variation, mode, iqr
from pyentrp import entropy as pent
import antropy as ent
from hurst import compute_Hc
import time

class Stats_features():

    def __init__(self, samples_num=1, stat_features=None):
        self.stat_features = stat_features
        self.samples_num = samples_num


    def crest_factor(self, data):
        return np.max(np.abs(data)) / np.sqrt(np.mean(np.square(data)))


    def hurst(self, data):
        H, c, _ = compute_Hc(data, kind='change')
        return H

    
    def hjorth(self, data):
        activity, complexity = ent.hjorth_params(data)
        return activity, complexity


    def energy(self, data):
        return sum(np.abs(data) ** 2)


    def range(self, data):
        return (max(data) - min(data))

    def stat_functions(self, data, everything, stat=None):
        if everything == 1:
            return {'variation': variation(data), 'kurtosis': kurtosis(data),\
            'shannon entropy': pent.shannon_entropy(data), 'range': self.range(data),\
            'iqr': iqr(data), 'sample entropy': ent.sample_entropy(data),\
            'skew': skew(data) , 'energy': self.energy(data), 'hurst': self.hurst(data),\
            'petrosian fd': ent.petrosian_fd(data), 'variance': data.var(),\
            'zero crossing': ent.num_zerocross(data), 'std': np.std(data),\
            'higuchi fd': ent.higuchi_fd(data), 'hjorth activity': self.hjorth(data)[0],\
            'hjorth complexity': self.hjorth(data)[1], 'crest factor': self.crest_factor(data),\
            'mean': np.mean(data), 'permutation entropy': ent.perm_entropy(data, normalize=True),\
            'svd entropy': ent.svd_entropy(data, normalize=True), 'approx entropy': ent.app_entropy(data),\
            'katz fd': ent.katz_fd(data), 'Detrended fluctuation analysis': ent.detrended_fluctuation(data)}
        
        else:
            return {'variation': variation(data), 'kurtosis': kurtosis(data),\
            'shannon entropy': pent.shannon_entropy(data), 'range': (max(data) - min(data)),\
            'iqr': iqr(data), 'sample entropy': float(pent.sample_entropy(data, 1)),\
            'skew': skew(data) , 'energy': self.energy(data), 'hurst': self.hurst(data),\
            'petrosian fd': ent.petrosian_fd(data), 'variance': data.var(),\
            'zero crossing': ent.num_zerocross(data), 'std': np.std(data),\
            'higuchi fd': ent.higuchi_fd(data), 'hjorth activity': self.hjorth(data)[0],\
            'hjorth complexity': self.hjorth(data)[1], 'crest factor': self.crest_factor(data),\
            'mean': np.mean(data), 'permutation entropy': ent.perm_entropy(data, normalize=True),\
            'svd entropy': ent.svd_entropy(data, normalize=True), 'approx entropy': ent.app_entropy(data),\
            'katz fd': ent.katz_fd(data), 'Detrended fluctuation analysis': ent.detrended_fluctuation(data)}.get(stat)

    def get_stat_features(self, data):
        
        if type(self.stat_features) == type(None):
        #     start_time = time.time()
            self.stat_features = self.stat_functions(data, 1).keys()
        # print(f"1 time: {time.time() - start_time} seconds ---")
        return self.stat_functions(data, 1)

        # else:
        #     start_time = time.time()
            
        #     features_dict = {}

        #     for feature in self.stat_features:
        #         features_dict[feature] = self.stat_functions(data, 0, feature)
        #     print(f"2 time: {time.time() - start_time} seconds ---")
        #     return features_dict


    def get_features_func(self, data, col_names=None):
        
        features_names = []
        
        if type(col_names) == list:
            col_names = col_names
        else:
            col_names = ['a2_x', 'a2_y', 'a2_z', 'label']

        extracted_features = pd.DataFrame()

        for bear_id in data['experiment_id'].unique():

            label = data[data['experiment_id'] == bear_id]['label'].unique()[0]

            exp_data = data[data['experiment_id'] == bear_id]

            for i in range(self.samples_num):
                
                exp_data_chunck = exp_data.iloc[
                    int(i * len(exp_data) / self.samples_num): int((i + 1) * len(exp_data) / self.samples_num)
                    ]
                features_row = ()

                for col in col_names:
                    
                    if col == 'label':

                        if not features_names:

                            for column in col_names[:-1]:
                                for stat in self.stat_features:
                                    features_names.append(column + ' ' + stat)
                            features_names.extend(['bear_id', 'label'])
                        
                        features_row = np.concatenate((features_row, int(bear_id), int(label)), axis=None)
                        features_row = pd.DataFrame(data=features_row).transpose().set_axis(features_names, axis=1)

                    else:
                        col_data = exp_data_chunck[col]
                        col_stat = list(self.get_stat_features(col_data).values())
                        features_row = np.concatenate((features_row, col_stat), axis=None)

                extracted_features = pd.concat([extracted_features, features_row])

        extracted_features = extracted_features.reset_index(drop=True)

        return extracted_features