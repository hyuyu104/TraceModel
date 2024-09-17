.. code:: ipython3

    import os
    import pandas as pd
    import numpy as np
    from scipy import stats
    import seaborn as sns
    from matplotlib import pyplot as plt
    import traceHMM.model as trm
    import traceHMM.utils.plot as tplt

.. code:: ipython3

    %reload_ext autoreload
    %autoreload 2

Notations
~~~~~~~~~

Let :math:`\mathbf X \in \mathbb R^{N\times T}` denote the spatial
distance between two loci. The :math:`nt`\ th entry of
:math:`\mathbf X`, denoted by :math:`X_t^n`, represents the spatial
distance of the :math:`n`\ th trace at timepoint :math:`t`. Define the
notation for hidden state similarly as
:math:`\mathbf H \in \mathbb R^{N\times T}` and :math:`H_t^n`. In this
section, we assume the Markov chain jumps between :math:`S` states, so
:math:`H_t^n \in \{1, ..., S\}`. In our paper, we assume :math:`S = 3`,
but the model works for arbitrary number of states. Further, we define
:math:`m_t^n \in \{1, 0\}` to indicate whether :math:`X_t^n` is observed
(:math:`m_t^n = 1` if :math:`X_t^n` is observed, otherwise :math:`0`).

We further define the PDF of spatial distance :math:`x` at state
:math:`s \in \{1, ..., S\}` as :math:`f_s(x)`. For the Markov chain,
define the initial distribution and transition probability as

.. math::


   \vec\mu = \begin{bmatrix}
   \mathbb P(H_1 = 1) \\ \cdots \\ \mathbb P(H_1 = S)
   \end{bmatrix}\qquad\text{and}\qquad
   \mathbf P = 
   \begin{bmatrix}
   \mathbb P(H_{t+1} = 1|H_t = 1) & \mathbb P(H_{t+1} = 2|H_t = 1) & \cdots \\
   \mathbb P(H_{t+1} = 1|H_t = 2) & \mathbb P(H_{t+1} = 2|H_t = 2) & \cdots \\
   \vdots & \vdots & \ddots
   \end{bmatrix}

We write :math:`\mathbb P(H_1 = s)` as :math:`\mu(s)` and
:math:`\mathbb P(H_{t+1} = s_2|H_t = s_1)` as :math:`P(s_1, s_2)` to
abbreviate the notations.

In our model, we assume :math:`f_s(x)` is known before fitting the HMM,
so the parameters of :math:`f_s(x)` are not updated. This allows
pre-specified values for the looped and unlooped status. For example,
one can fix the loop state mean to :math:`200`\ nm to ensure consistent
prediction of loop state across different datasets. In the end, the
fitting procedure described below is used to determine :math:`\vec\mu`
and :math:`\mathbf P` only. This reduction of estimable parameters also
facilitates stable estimation across different runs.

In addition, we allow some entries of the transition matrix to be fixed.
For example, one can fix :math:`P(1, 3) = 0`, which means the loop state
cannot jump directly to the unloop state but must pass through the
intermediate state first. This additional flexibility makes it possible
to incorporate more biological background to the model, thus improving
prediction results.

Test by Simulations
-------------------

.. code:: ipython3

    P = np.array([
        [0.95, 0.05,    0],
        [0.02, 0.96, 0.02],
        [   0, 0.05, 0.95]
    ])
    err = np.diag(np.square([0.06, 0.06, .12])*2)
    dist_params = (
            {"cov":np.diag(np.ones(3)*0.015), "err":err},
            {"cov":np.diag(np.ones(3)*0.055), "err":err},
            {"cov":np.diag(np.ones(3)*0.085), "err":err}
    )
    tse = trm.TraceSimulator(
        P=P,
        mu=np.array([1/3, 1/3, 1/3]),
        dist_params=dist_params,
        dist_type=trm.multivariate_normal,
        random_state=100
    )
    H, X0 = tse.simulate_multiple_traces(500, 400)
    X = tse.mask_by_markov_chain(X0, 0.5, a=0.8)


.. parsed-literal::

    P(stay observed) = 0.8


.. code:: ipython3

    dist_params = (
            {"cov":np.diag(np.ones(3)*0.015)},
            {"cov":np.diag(np.ones(3)*0.055)},
            {"cov":np.diag(np.ones(3)*0.085)}
    )
    tm = trm.TraceModel(
        X=X, Pm=np.array([
            [-1, -1,  0],
            [-1, -1, -1],
            [ 0, -1, -1]
        ]), 
        dist_params=dist_params, 
        dist_type=trm.multivariate_normal, 
        update_dist_params=["err"]
    )
    tm.fit(600)


.. parsed-literal::

    Converged at iteration 504


.. code:: ipython3

    tm.loc_err




.. parsed-literal::

    array([0.05237644, 0.05454736, 0.11672494])



.. code:: ipython3

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.scatterplot(tm.convergence, ax=axes[0])
    sns.scatterplot(tm.lklhd, ax=axes[1])




.. parsed-literal::

    <Axes: >




.. image:: simulations_files/simulations_7_1.png


.. code:: ipython3

    fig = tplt.plot_transition_matrix(tm.P)
    fig = tplt.plot_transition_matrix(P)



.. image:: simulations_files/simulations_8_0.png



.. image:: simulations_files/simulations_8_1.png


.. code:: ipython3

    n = 1
    dist = np.linalg.norm(X[n], axis=1)
    df = pd.DataFrame({"dist":dist, "state":tm.decode(X[[n]])[0]})
    df = df.reset_index(names="t")
    df["true"] = H[n]
    code_book = {0:"looped", 1:"intermediate", 2:"unlooped"}
    fig, axes = plt.subplots(2, 1, figsize=(16, 6))
    tplt.plot_trace(df, "t", "dist", "state", code_book, fig, axes[0])
    tplt.plot_trace(df, "t", "dist", "true", code_book, fig, axes[1])
    axes[0].set(xlabel="Time (s)", ylabel="Spatial distance (µm)", title="Predicted loop states")
    axes[1].set(xlabel="Time (s)", ylabel="Spatial distance (µm)", title="True loop states")
    fig.tight_layout()



.. image:: simulations_files/simulations_9_0.png


Ignore the localization error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    dist_params = (
            {"cov":np.diag(np.ones(3)*0.015)},
            {"cov":np.diag(np.ones(3)*0.055)},
            {"cov":np.diag(np.ones(3)*0.085)}
    )
    tm2 = trm.TraceModel(
        X=X, Pm=np.array([
            [-1, -1,  0],
            [-1, -1, -1],
            [ 0, -1, -1]
        ]), 
        dist_params=dist_params, 
        dist_type=trm.multivariate_normal, 
    )
    tm2.fit(600)


.. parsed-literal::

    Converged at iteration 224


.. code:: ipython3

    fig = tplt.plot_transition_matrix(tm2.P)



.. image:: simulations_files/simulations_12_0.png


.. code:: ipython3

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.scatterplot(tm2.convergence, ax=axes[0])
    sns.scatterplot(tm2.lklhd, ax=axes[1])




.. parsed-literal::

    <Axes: >




.. image:: simulations_files/simulations_13_1.png


.. code:: ipython3

    n = 1
    dist = np.linalg.norm(X[n], axis=1)
    df = pd.DataFrame({"dist":dist, "state":tm2.decode(X[[n]])[0]})
    df = df.reset_index(names="t")
    df["true"] = H[n]
    code_book = {0:"looped", 1:"intermediate", 2:"unlooped"}
    fig, axes = plt.subplots(2, 1, figsize=(16, 6))
    tplt.plot_trace(df, "t", "dist", "state", code_book, fig, axes[0])
    tplt.plot_trace(df, "t", "dist", "true", code_book, fig, axes[1])
    axes[0].set(xlabel="Time (s)", ylabel="Spatial distance (µm)", title="Predicted loop states")
    axes[1].set(xlabel="Time (s)", ylabel="Spatial distance (µm)", title="True loop states")
    fig.tight_layout()



.. image:: simulations_files/simulations_14_0.png


