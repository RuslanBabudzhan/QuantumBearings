import os
import configparser
from typing import List, Tuple, Optional, Dict, Union

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from source.datamodels.datamodels import BootstrapResults, BaseResultsData
from source.utils import get_project_root


def dist_plot(results: Union[List[BootstrapResults], List[Dict[str, List[float]]], List[Dict[str, np.ndarray]]],
              models: List[str],
              plot_type: str,
              metric: str,
              title: str,
              to_png: bool = False,
              use_config=True,
              filename: Optional[str] = None,
              filepath: Optional[str] = ''):
    """
    Plot metric distributions
    :param results: list of results or dict
    :param models: model names corresponding to each item in results
    :param plot_type: type of plot. Can be kdeplot or boxenplot
    :param metric: metric to select from results
    :param title: plot title
    :param to_png: Whether the plot should be saved as *.png
    :param use_config: Use userconfig.ini in project root dir to customize the plot.
    :param filename: name of the image
    :param filepath: path to the image
    :return: None
    """
    if to_png and not filename:
        raise ValueError('Got no filename. to_png option is permitted')

    plots = {'kdeplot': sns.kdeplot,
             'boxenplot': sns.boxenplot}

    if plot_type not in plots.keys():
        raise ValueError(f'Incorrect plot type. Got {plot_type}. Expected {set(plots.keys())}')

    if use_config:
        root = get_project_root()
        config = configparser.ConfigParser()
        config.read(os.path.join(root, "userconfig.ini"))
        plot_size = (float(config['Plots']['width']), float(config['Plots']['height']))
        font_scale = float(config['Plots']['font_scale'])
        dpi = int(config['Plots']['dpi'])
        style = config['Plots']['style']
        palette = config['Plots']['palette']
    else:
        plot_size = (6.4, 4.8)
        font_scale = 1.5
        dpi = 100
        style = 'whitegrid'
        palette = 'rainbow'
    if isinstance(results[0], BootstrapResults):
        scores = [result.bootstrap_scores for result in results]
    else:
        scores = [result for result in results]

    data_to_plot = np.array(list(map(lambda vector: vector[metric], scores))).T
    results_df = pd.DataFrame(data_to_plot).set_axis(models, axis=1)

    sns.set_theme(font_scale=font_scale)
    palette_obj = sns.color_palette(palette=palette, n_colors=len(models))
    plt.figure(figsize=plot_size, dpi=dpi).suptitle(title)
    sns.set_style(style)
    plot = plots[plot_type](data=results_df, palette=palette_obj)
    if to_png:
        path = os.path.join(filepath, filename)
        plot.get_figure().savefig(path)
    plt.show()


def bar_plot(results: Union[List[BaseResultsData], List[Dict[str, float]]],
             models: List[str],
             metrics: List[str],
             title: str,
             plot_vals: bool = False,
             to_png: bool = False,
             use_config=True,
             filename: Optional[str] = None,
             filepath: Optional[str] = ''):
    """
    Plot metric values
    :param results: list of results or dict
    :param models: model names corresponding to each item in results
    :param metrics: metrics to select from results
    :param title: plot title
    :param plot_vals: Whether metric values should be plotted on bars
    :param to_png: Whether the plot should be saved as *.png
    :param use_config: Use userconfig.ini in project root dir to customize the plot.
    :param filename: name of the image
    :param filepath: path to the image
    :return: None
    """
    if to_png and not filename:
        raise ValueError('Got no filename. to_png option is permitted')

    if use_config:
        root = get_project_root()
        config = configparser.ConfigParser()
        config.read(os.path.join(root, "userconfig.ini"))
        plot_size = (float(config['Plots']['width']), float(config['Plots']['height']))
        font_scale = float(config['Plots']['font_scale'])
        dpi = int(config['Plots']['dpi'])
        style = config['Plots']['style']
        palette = config['Plots']['palette']
    else:
        plot_size = (6.4, 4.8)
        font_scale = 1.5
        dpi = 100
        style = 'whitegrid'
        palette = 'rainbow'

    if isinstance(results[0], BootstrapResults):
        scores_list = [result.scores for result in results]
    else:
        scores_list = [result for result in results]

    scores = np.array(list(map(lambda n: list(n.values()), scores_list)))
    len_scores = scores.shape[0] * scores.shape[1]
    scores_df = pd.Series(scores.reshape(len_scores), name='scores')
    metrics_df = pd.Series(
        np.array(list(
            map(lambda n: list(n.keys()), scores_list)
        )).reshape(len_scores),
        name='metrics')

    models_df = pd.Series(models, name='models').repeat(int(len(metrics_df) / len(models))).reset_index(drop=True)

    results_df = pd.concat([scores_df, metrics_df, models_df], axis=1)
    results_df = results_df[results_df['metrics'].isin(metrics)]

    sns.set(font_scale=font_scale)
    plt.figure(figsize=plot_size, dpi=dpi).suptitle(title)
    sns.set_style(style)
    palette_obj = sns.color_palette(palette=palette, n_colors=len(metrics))
    plot = sns.barplot(x='models', y='scores', hue='metrics', data=results_df, palette=palette_obj)

    if plot_vals:
        for p in plot.patches:
            plot.annotate(format(p.get_height(), '.2f'),
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center',
                          size=15,
                          xytext=(0, -12),
                          textcoords='offset points')

    if to_png:
        path = os.path.join(filepath, filename)
        plot.get_figure().savefig(path)
    plt.show()












