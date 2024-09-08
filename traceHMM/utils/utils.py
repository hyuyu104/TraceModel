import warnings
import numpy as np
import pandas as pd


def ffill(
        data:pd.DataFrame,
        tracj_id_col:str,
        time_col:str,
        dist_col:str,
        method:str="ffill",
        limit:int=None
) -> pd.DataFrame:
    """Fill missing spatial distance with the immediate available value that
    precedes it.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe with missing rows or rows with NaNs
    tracj_id_col : str
        The column name of trajectory id
    time_col : str
        The column name of time
    dist_col : str
        The column name of spatial distance
    method : str, optional
        Tilling method, by default "ffill". Passed to pd.reindex function.
    limit : int, optional
        How many values before the missing are checked, by default None. 
        Passed to pd.reindex function.

    Returns
    -------
    pd.DataFrame
        Output dataframe with all missing spatial distance filled by their
        preceding values.
    """
    frame_df = data.set_index(time_col)
    # make sure rows with no available distance are dropped
    frame_df = frame_df.dropna(subset=[dist_col])
    def func(df):
        t_range = np.arange(np.ptp(df.index.values)+1)
        t_range += np.min(df.index)
        return df.reindex(t_range, method=method, limit=limit)
    filled = frame_df.groupby(
        tracj_id_col, 
        sort=False
    ).apply(func, include_groups=False).reset_index()
    return filled


def stationary_dist(P:np.ndarray) -> np.ndarray:
    """Find the stationary distribution of a transition matrix.
    Calculate the eigenvector of P' with corresponding eigenvalue being one.

    Parameters
    ----------
    P : np.ndarray
        input transition matrix

    Returns
    -------
    np.ndarray
        stationary distribution
    """
    L, V = np.linalg.eig(P.T)
    v = V[:, np.abs(L - 1) < 1e-10]
    if v.shape[1] > 1:
        warnings.warn(UserWarning(
            "Multiple stationary distributions. " +
            "Return the first one."
        ))
        v = v[:, 0]
    v = (v/np.sum(v)).flatten()
    return v