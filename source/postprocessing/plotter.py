import os
from typing import List, Tuple, Optional, Dict, Union
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from source.datamodels.datamodels import BootstrapResults, BaseResultsData


def dist_plot(results: Union[BootstrapResults, Dict[str, list]],
              models: List[str],
              plot_type: str,
              metrics: List[str],
              plot_size: Tuple[float, float],
              Title: str,
              filename: str,
              filepath: Optional[str]=''):

    if isinstance(results, BootstrapResults):
        scores_dict = results.bootstrap_scores
    else: 
        scores_dict = results

    results_df = pd.DataFrame(list(map(lambda n: n[metrics], scores_dict))).T.set_axis(models, axis=1)

    plots = {'kdeplot': sns.kdeplot,
             'boxenplot': sns.boxenplot}

    if plot_type in plots.keys():
        sns.set(font_scale=1.5)
        plt.figure(figsize=plot_size).suptitle(Title)
        sns.set_style("darkgrid")
        plot = plots[plot_type](data=results_df)
        path = os.path.join(filepath, filename)
        plot.get_figure().savefig(path) 
        plt.show()


def bar_plot(results: Union[BaseResultsData, List[Dict[str, float]]],
              models: List[str],
              metrics: List[str],
              plot_size: Tuple[float, float],
              Title: str,
              filename: str,
              filepath: Optional[str]=''):
    
    if isinstance(results, BaseResultsData):
        scores_dict = results.scores
    else: 
        scores_dict = results  

    scores = np.array(list(map(lambda n: list(n.values()), scores_dict)))  
    len_scores = scores.shape[0] * scores.shape[1]
    scores_df = pd.Series(scores.reshape(len_scores), name='scores')
    metrics_df = pd.Series(
        np.array(list(
            map(lambda n: list(n.keys()), scores_dict)
                )).reshape(len_scores), 
                name = 'metrics')
    models_df = pd.Series(models, name='models').repeat(len(metrics_df)/len(models)).reset_index(drop=True)
    
    results_df = pd.concat([scores_df, metrics_df, models_df], axis=1)
    results_df = results_df[results_df['metrics'].isin(metrics)]

    sns.set(font_scale=1.5)
    plt.figure(figsize=plot_size).suptitle(Title)
    sns.set_style("darkgrid")
    plot = sns.barplot(x='models', y='scores', hue='metrics', data=results_df)

    for p in plot.patches:
        plot.annotate(format(p.get_height(), '.2f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    size=15,
                    xytext = (0, -12), 
                    textcoords = 'offset points')

    path = os.path.join(filepath, filename)
    plot.get_figure().savefig(path) 
    plt.show()