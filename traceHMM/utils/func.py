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
        Input dataframe with missing rows or rows with NaNs.
    tracj_id_col : str
        The column name of trajectory id.
    time_col : str
        The column name of time.
    dist_col : str
        The column name of spatial distance.
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
    """Find the stationary distribution of a transition matrix. Calculate the 
    eigenvector of P' with corresponding eigenvalue being one.

    Parameters
    ----------
    P : (S, S) np.ndarray
        Input transition matrix.

    Returns
    -------
    (S) np.ndarray
        Stationary distribution.
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


def long_to_tensor(
    df:pd.DataFrame,
    id_col:str,
    t_col:str,
    val_cols:str|list
) -> np.ndarray:
    """Generate the input tensor for the TraceModel class.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe in the long format: each row is a time point for one
        trace. Has time, trace ID, and values (spatial distance/difference in
        each axis/raw coordinates) as columns.
    id_col : str
        Trace ID column name.
    t_col : str
        Time column name.
    val_cols : str | list
        Either a list of value column names or a single value column name.

    Returns
    -------
    (N, T, D) np.ndarray
        N is the total number of traces (number of unique trace IDs), T is 
        the total number of time points, and D is the length of `val_cols`.
    """
    if isinstance(val_cols, str):
        val_cols = [val_cols]
    pivoted = df.pivot(index=t_col, columns=id_col, values=val_cols)
    arr = np.stack([pivoted[t] for t in val_cols])
    return arr.transpose((2, 1, 0))


def add_predictions_to_df(
    df:pd.DataFrame,
    decoded_states:np.ndarray,
    X:np.ndarray=None,
    id_col:str=None,
    t_col:str=None,
    val_cols:str|list=None,
    num_name:str="state",
    code_book:dict=None,
    loop_name:str="loop"
):
    """Add decoding results to the dataframe. Must provide `id_col`, `t_col`, 
    and `val_cols` if `X`, the tensor format of `df`, is not supplied.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to add the decoding results.
    decoded_states : np.ndarray
        Decoding results from TraceModel.decode.
    X : np.ndarray, optional
        Obtained by running long_to_tensor on `df`, by default None.
    id_col : str
        Trace ID column name.
    t_col : str
        Time column name.
    val_cols : str | list
        Either a list of value column names or a single value column name.
    num_name : str, optional
        Decoding state column name, by default "state".
    code_book : dict, optional
        A dictionary with keys as integers (states) and values as strings (
        name of each state), by default None.
    loop_name : str, optional
        If `code_book` is not None, use this as the column name for the names
        of each state, by default "loop".

    Raises
    ------
    ValueError
        If `decoded_states` cannot be reshaped to match the shape of `df`.
    """
    if X is None:
        X = long_to_tensor(df, id_col, t_col, val_cols)
        
    avail = ~np.isnan(X[...,0])
    # case where df already has missing rows filled with NaN
    if decoded_states.size == len(df):
        df[num_name] = decoded_states.flatten().astype("int")
    elif np.sum(avail) == len(df):
        df[num_name] = decoded_states[avail].astype("int")
    else:
        raise ValueError("Inconsistent shape.")

    if code_book is not None:
        df[loop_name] = df[num_name].map(code_book)
        
        
def avg_loop_life_time(H:np.ndarray) -> float:
    """Calculate the mean loop life time of (a) given sequence(s) of hidden 
    states (not the observed values). Loop state is denoted by 0 in the input
    sequence(s).

    Parameters
    ----------
    H : (T) or (N, T) np.ndarray
        A single sequence of hidden states or N sequences.

    Returns
    -------
    float
        The average loop life time of the sequence.
    """
    if len(H.shape) == 1:
        H = [H]
    loops = []
    for h in H:
        added = False
        for t, val in enumerate(h):
            if val == 0:
                if added:
                    loops[-1] += 1
                else:
                    loops.append(1)
                    added = True
            else:
                added = False
    return np.mean(loops)