import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import matplotlib as mpl
# remove the top and right spines
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
# remove the border of legend
mpl.rcParams["legend.frameon"] = False
mpl.rcParams["legend.loc"] = (1, 0.5)

def plot_trace(
        df:pd.DataFrame, 
        t_col:str, 
        d_col:str, 
        l_col:str=None, 
        states:dict=None,
        fig:plt.figure=None,
        ax:plt.axes=None
) -> tuple:
    """Plot a single trace along with its looping status if available.

    Parameters
    ----------
    df : pd.DataFrame
        data frame in long format
    t_col : str
        name of the time column
    d_col : str
        name of the distance column
    l_col : str, optional
        name of the loop status column, column values are integers, by default None
    states : dict, optional
        code book for loop status, must be included if l_col is not None, by default None
    fig : plt.figure, optional
        fig object, by default None
    ax : plt.axes, optional
        ax object, by default None

    Returns
    -------
    tuple
        (fig, ax)
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
    sns.lineplot(df, x=t_col, y=d_col, c="green", ax=ax)
    if l_col is None:
        sns.scatterplot(df, x=t_col, y=d_col, s=40, c="green", ax=ax)
    else:
        df = df.copy()
        df["state"] = df[l_col].map(states)
        palette = {m:sns.color_palette("tab10")[i] for i,m in states.items()}
        sns.scatterplot(
            df, x=t_col, y=d_col, hue="state", 
            style="state", palette=palette, s=80, ax=ax,
            style_order=states.values()
        )
        ax.legend(loc="upper right")

    ax.spines["right"].set_visible(True)
    return fig, ax


def plot_transition_matrix(transmat_):
    fig, (ax1, ax2) = plt.subplots(ncols=2, width_ratios=(4, 1), figsize=(5, 3))
    labs = ["looped", "intermediate", "unlooped"]
    sns.heatmap(
        transmat_, 
        square=True, 
        cbar=False,
        xticklabels="",
        yticklabels=labs,
        annot=transmat_,
        ax=ax1
    )
    nstates = len(transmat_)
    stationary_dist, prev = np.ones(nstates)/nstates, None
    while prev is None or np.linalg.norm(prev - stationary_dist) > 1e-4:
        prev = stationary_dist
        stationary_dist = stationary_dist @ transmat_
    sns.heatmap(
        stationary_dist[:,None], 
        cbar=False, 
        annot=True,
        xticklabels="",
        yticklabels="",
        ax=ax2,
    )
    ax2.set_ylabel("Stationary distribution")
    return fig