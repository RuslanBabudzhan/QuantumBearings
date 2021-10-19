import pandas as pd
import numpy as np
import time
from typing import Union
import sklearn

class Features_selection:

    def __init__(self, n=10):
        self.n_features = n # number of selected features


    def VT_export(self, 
                  fs_method: sklearn.base.BaseEstimator, 
                  X: Union[pd.DataFrame, np.ndarray], 
                  y: Union[pd.DataFrame, np.ndarray]) -> list:
        """
        fs_model: sklearn.feature_selection.VarianceThreshold
        
        Fuction takes X and y data, fit VarianceThreshold object, reterns df with selected features.
        """
        fs_method.fit(X, y)
        important_features = pd.DataFrame(
            [fs_method.feature_names_in_, np.zeros(len(fs_method.feature_names_in_))]).\
                transpose().\
                    rename(columns={0: 'feature'})
        return important_features['feature'].iloc[:self.n_features].to_list()


    def wrapper_fs_export(self, 
                          fs_method: sklearn.base.BaseEstimator, 
                          X: Union[pd.DataFrame, np.ndarray], 
                          y: Union[pd.DataFrame, np.ndarray]) -> list:
        """
        fs_model:
            sklearn.feature_selection.SequentialFeatureSelector
            sklearn.feature_selection.RFE
            sklearn.feature_selection.SelectFromModel
        
        Fuction takes X and y data, fit wrapper methods such Recursive Feature Elimination, 
        Sequential Feature Selection or Select From Model, reterns df with selected features.
        """
        fs_method.fit(X, y)

        try:
            importance = fs_method.ranking_
        except AttributeError:
            importance = np.zeros(self.n_features)

        important_features = pd.DataFrame(
            [fs_method.feature_names_in_, importance]).\
                transpose().\
                    rename(columns={0: 'feature', 1: 'importance'}).\
                        sort_values(by='importance', ignore_index=True)
        
        return important_features['feature'].iloc[:self.n_features].to_list()


    def filter_method_export(self, 
                             fs_method: sklearn.base.BaseEstimator,    
                             X: Union[pd.DataFrame, np.ndarray], 
                             y: Union[pd.DataFrame, np.ndarray]) -> list:
        """
        fs_model:
            sklearn.feature_selection.SelectFpr
            sklearn.feature_selection.feature_selection.SelectPercentile
            sklearn.feature_selection.SelectFdr
            sklearn.feature_selection.SelectFwe
        
        Fuction takes X and y data, fit filter methods object, reterns df with selected features.
        """
        fs_method.fit_transform(abs(X), abs(y))
        important_features = pd.DataFrame(
            [fs_method.feature_names_in_, fs_method.pvalues_]).\
                transpose().\
                    rename(columns={0: 'feature', 1: 'importance'}).\
                        sort_values(by='importance', ignore_index=True)
        
        return important_features['feature'].iloc[:self.n_features].to_list()


    def select_with_method(self, 
                           methods: dict, 
                           X: Union[pd.DataFrame, np.ndarray], 
                           y: Union[pd.DataFrame, np.ndarray]):
        """
        Iterate by FS dict and return selected features as DataFrame
        """
        important_features_df = pd.DataFrame([])

        for name, method in zip(methods.keys(), methods.values()):

            start_time = time.time()

            if name == 'VT':
                important_features = self.VT_export(method, X.values, y)

            elif any(mod in name for mod in ['RFE', 'SFM', 'SFS']):
                important_features = self.wrapper_fs_export(method, X.values, y)

            else:
                important_features = self.filter_method_export(method, X.values, y)
            
            print(f"{name} --- time: {time.time() - start_time} seconds ---")

            important_features_df[name] = important_features

        return important_features_df