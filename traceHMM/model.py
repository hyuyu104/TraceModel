""" The TraceModel class and a simulator for Markov chains.
"""

from collections import deque
from itertools import product
import numpy as np
from scipy import stats
from .update import (
    scaled_forward, 
    scaled_backward,
    calculate_v,
    log_viterbi
) # type: ignore

class TraceModel:
    """Modified hidden Markov model to infer chromatin loops.
    
        Parameters
        ----------
        X : (N, T, D) np.ndarray
            Input value (of D dimension) of N trajectories over T time points.
        Pm : (S, S) np.ndarray
            Initial transition matrix. Entries with negative values will be 
            updated. Other entries with nonnegative values will not be updated 
            and will be the same after fitting.
        dist_params : tuple[dict, ...]
            Tuple of length S. Each element is a dictionary specifying the 
            parameters of the distribution at each state.
        dist_type : distribution class, optional
            A distribution class that has a `pdf` method, by default stats.norm. 
            `dist_params` will be passed in as keyword arguments.
        update_dist_params : list, optional
            A list of parameters to update for the distributions at each state,
            by default None. If `update_dist_params` is not None, then the 
            distribution class supplied must have a `mle` method that returns a 
            dictionary with keys containing update_dist_params.
    """
    def __init__(
            self, 
            X:np.ndarray,
            Pm:np.ndarray,
            dist_params:tuple[dict, ...],
            dist_type=stats.norm,
            update_dist_params:list=None,
    ):
        self._N, self._T, self._D = X.shape
        self._S = Pm.shape[0]
        self._dist_params = dist_params
        self._dist_type = dist_type
        self._X = X
        self._convergence = []
        self._lklhd = []
        self._m = np.where(Pm < 0)
        self._update_dist_params = update_dist_params

        # expect row sum except the fixed entries
        self._l = 1 - np.sum(np.where(Pm < 0, 0, Pm), axis=1)
        initP = self._l/np.sum(Pm<0, axis=1)
        initP = np.repeat(initP[:,None], self._S, axis=1)
        # initialize transition matrix
        self._P = np.where(Pm < 0, initP, Pm)

        # initialize initial distribution
        self._mu = np.ones(self._S)/self._S

    @property
    def P(self) -> np.ndarray:
        """Transition probability matrix."""
        return self._P
    
    @P.setter
    def P(self, value:np.ndarray):
        """
        Set transition matrix manually, without changing fixed entries 
        specified by `Pm`.
        """
        P = self._P.copy()
        P[self._m] = value[self._m]
        if np.any(np.abs(np.sum(P, axis=1) - 1) > 1e-6):
            raise ValueError(
                "Row sum is not one. Note fixed entries are not updated."
            )
        self._P = P
    
    @property
    def mu(self) -> np.ndarray:
        """Initial distribution. Initialized to be uniform in each state."""
        return self._mu
    
    @mu.setter
    def mu(self, value:np.ndarray):
        """Set initial distribution manually."""
        self._mu = value
    
    @property
    def N(self) -> int:
        """Number of trajectories in `X`."""
        return self._N
    
    @property
    def T(self) -> int:
        """Number of time points in `X`."""
        return self._T
    
    @property
    def S(self) -> int:
        """Number of states."""
        return self._S
    
    @property
    def convergence(self) -> list:
        """The mean absolute difference between the updated transition matrix 
        and the transition matrix from the last iteration.
        """
        return self._convergence
    
    @property
    def lklhd(self) -> list:
        """A list of the log-likelihood of the model at each iteration.
        """
        return self._lklhd
    
    @property
    def loc_err(self) -> np.ndarray:
        """The estimated localization error in nm/um. Available only if 
        update_dist_params contains "err".
        """
        return np.sqrt(np.diag(self._dist_params[0]["err"])/2)

    def density(
            self, 
            state:int|np.ndarray, 
            x:np.ndarray
    ) -> int|np.ndarray:
        """Calculates the probability of given observation(s) at state(s).

        Parameters
        ----------
        state : int | np.ndarray
            A single state or an array of states to evaluate x.
        x : (..., D) np.ndarray
            Observations. Can be (T, D), which specifies the observations at T
            time points, or (N, T, D), which specifies the observations at T
            time points of N observations.

        Returns
        -------
        int | np.ndarray
            Probability of observing x at state(s). Will broadcast.
        """
        if np.issubdtype(type(state), np.integer):
            args = self._dist_params[state]
            vals = self._dist_type.pdf(x, **args)
            if len(x.shape) == 1:
                if np.isnan(vals):
                    return 1
                return vals
        else:
            vals = np.stack([
                self._dist_type.pdf(x, **self._dist_params[s])
                for s in state
            ])
        return np.where(np.isnan(vals), 1, vals)
    
    def _forward_(self, X:np.ndarray):
        den = self.density(np.arange(self._S), X)
        sf = scaled_forward(den, self._mu, self._P, self._N, self._T, self._S)
        self._alpha = sf[:,:,-1]
        self._f = sf[:,:,:-1]

    def _backward_(self, X:np.ndarray):
        den = self.density(np.arange(self._S), X)
        sb = scaled_backward(den, self._P, self._N, self._T, self._S)
        self._b = sb

    def _calculate_u_(self):
        u_raw = self._f*self._b
        u_sum = np.sum(u_raw, axis=2)
        u = np.stack([a/b[:,None] for a, b in zip(u_raw, u_sum)])
        self._u = u

    def _calculate_v_(self, X:np.ndarray):
        self._v = calculate_v(
            self.density(np.arange(self._S), X),
            self._f, self._b, self._alpha, 
            self._P,
            self._N, self._T, self._S
        )

    def _update_params_(self):
        # update the initial distribution
        self._mu = np.mean(self._u[:,0,:], axis=0)

        # update transition probabilities
        P_hat = np.zeros_like(self._P)
        states = np.arange(self._S)
        for s1, s2 in product(states, states):
            a = np.sum(self._v[:,1:,s1,s2])
            
            # select entries that need to be updated
            s2s = self._m[1][self._m[0]==s1]
            b = np.sum(self._v[:,1:,s1,s2s])

            P_hat[s1,s2] = self._l[s1]*a/b
            
        # update distribution parameters, if required
        if self._update_dist_params is not None:
            new_params = self._dist_type.mle(self)
            for param in self._update_dist_params:
                for i in range(len(self._dist_params)):
                    self._dist_params[i][param] = new_params[param][i]

        self._convergence.append(np.mean(np.abs(P_hat - self._P)[self._m]))
        # update P after calculating the mean absolute error
        self._P[self._m] = P_hat[self._m]

    def fit(self, max_iter:int=100, cutoff:float=1e-4):
        """Fit the TraceModel using the input data.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations, by default 100
        cutoff : float, optional
            Criterion to terminate iterations, by default 1e-4. Fitting will 
            end if the mean absolute difference of transition matrix is below 
            this cutoff.
        """
        for _ in range(max_iter):
            self._forward_(self._X)
            self._lklhd.append(np.sum(np.log(self._alpha)))
            self._backward_(self._X)

            self._calculate_u_()
            self._calculate_v_(self._X)

            self._update_params_()
            if self._convergence[-1] < cutoff:
                print(f"Converged at iteration {len(self._convergence)}")
                break

    def _viterbi_(self, x:np.ndarray) -> np.ndarray:
        den = self.density(np.arange(self._S), x)
        decoded_states = log_viterbi(den, self._mu, self._P, len(x), self.S)
        return decoded_states
    
    def decode(self, X:np.ndarray=None) -> np.ndarray:
        """Predict the looping status of the input data using the fitted 
        model. Has to be run after `fit`.

        Parameters
        ----------
        X : (N', T', D) np.ndarray, optional
            New chromatin trace to predict looping status, by default None. If
            None is passed, will return the predicted loop status of the input
            used to train the model.

        Returns
        -------
        np.ndarray
            Same shape as `X`. Contains the prediction result.
        """
        X = self._X if X is None else X
        return np.stack([self._viterbi_(x) for x in X])
    

class TraceSimulator:
    """Simulate chromatin traces based on a Markov chain.
    
        Parameters
        ----------
        P : (S, S) np.ndarray
            The transition probability of the Markov chain.
        mu : (S) np.ndarray
            The initial distribution.
        dist_params : tuple[dict, ...]
            Tuple of length S. Each element is a dictionary specifying the 
            parameters of the distribution at each state.
        dist_type : _type_, optional
            A distribution class that has a `rvs` method, by default stats.norm. 
            `dist_params` will be passed in as keyword arguments.
    """
    def __init__(
            self,
            P:np.ndarray,
            mu:np.ndarray,
            dist_params:tuple[dict, ...],
            dist_type=stats.norm,
            random_state=None
    ):
        self._P = P
        self._mu = mu
        self._dist_params = dist_params
        self._dist_type = dist_type
        if random_state is not None:
            np.random.seed(random_state)
        
    @property
    def P(self) -> np.ndarray:
        """Transition probability matrix."""
        return self._P
    
    @property
    def mu(self) -> np.ndarray:
        """Initial distribution. Initialized to be uniform in each state."""
        return self._mu

    def simulate_single_trace(
            self, T:int
        ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate a single trace based on the input transition matrix and
        initial distribution.

        Parameters
        ----------
        T : int
            Number of time points to generate.

        Returns
        -------
        tuple[(T) np.ndarray, (T, D) np.ndarray]
            The first element is the true loop status of the trace. The second
            element is the spatial distance generated at each time point.
        """
        H = [np.random.choice(len(self._mu), p=self._mu)]
        for t in range(1, T):
            H.append(np.random.choice(len(self._mu), p=self._P[H[-1]]))
        H = np.array(H)
        
        xs = []
        for s in range(len(self._mu)):
            args = self._dist_params[s]
            xs.append(self._dist_type.rvs(**args, size=T))
        X = np.zeros_like(xs[-1])
        for s in range(len(self._mu)):
            X[np.where(H==s)] = xs[s][np.where(H==s)]
        return H, X
    
    def simulate_multiple_traces(
            self, T:int, N:int
        ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate multiple traces based on the input transition matrix and
        initial distribution.

        Parameters
        ----------
        T : int
            Number of time points to generate.
        N : int
            Number of traces to generate.

        Returns
        -------
        tuple[(N, T) np.ndarray, (N, T) np.ndarray]
            The first element is the true loop status of the trace. The second
            element is the spatial distance generated at each time point.
        """
        H, X = [], []
        for _ in range(N):
            h, x = self.simulate_single_trace(T)
            H.append(h)
            X.append(x)
        H = np.stack(H)
        X = np.stack(X)
        return H, X
    
    @staticmethod
    def mask_spatial_distance(
        self, X:np.ndarray, p_obs:float
    ) -> np.ndarray:
        """Mask spatial distance uniformly at random to simulate the missing
        data problem observed in real imaging data.

        Parameters
        ----------
        X : (..., T, D) np.ndarray
            Input spatial distance to be masked.
        p_obs : float
            The observed probability.

        Returns
        -------
        (..., T) np.ndarray
            Same shape as `X` with missing observations filled by NaN.
        """
        mask = np.random.choice(2, size=X.shape, p=[1 - p_obs, p_obs])
        masked_X = X.copy()
        masked_X[np.where(mask==0)] = np.nan
        return masked_X
    
    def mask_by_markov_chain(
        self, X:np.ndarray, p_obs:float, a:float=0.8
    ) -> np.ndarray:
        """Mask spatial distance by a two-state Markov chain. This is inspired 
        by the fact that missing observations are typically continuous in real
        imaging data.

        Parameters
        ----------
        X : (..., T, D) np.ndarray
            Input spatial distance to be masked.
        p_obs : float
            The observed probability.
        a : float, optional
            P(stay unobserved), by default 0.8. Needed as otherwise the linear
            system resulting from stationary distribution has infinite 
            solutions.

        Returns
        -------
        (..., T) np.ndarray
            Same shape as `X` with missing observations filled by NaN.
        """
        b = 1 - (1 - p_obs)*(1 - a)/p_obs
        print(f"P(stay observed) = {round(b, 3)}")
        P = np.array([
            [a, 1 - a],
            [1 - b, b]
        ])
        mu = np.array([1 - p_obs, p_obs])
        mc = TraceSimulator(
            P, mu, self._dist_params, self._dist_type
        )
        mask = mc.simulate_multiple_traces(
            X.shape[1], X.shape[0]
        )[0]
        masked_X = X.copy()
        masked_X[np.where(mask==0)] = np.nan
        return masked_X


class multivariate_chisq:
    """Multivariate independent chi-square distribution."""
    def pdf(X:np.ndarray, scales:np.ndarray) -> float|np.ndarray:
        """Independent scaled Chi-squared distributions each with a degree of 
        freedom of one.

        Parameters
        ----------
        X : np.ndarray
            Single observations or observations in ndarrays. The last axis has
            the same dimension as the length of scales.
        scales : np.ndarray
            The scale of each Chi-squared distribution. Assume the observation 
            is $Y^2$, where $Y = s X$ with $X ~ N(0, 1)$. The scale is 
            $s^2$.

        Returns
        -------
        float|np.ndarray
            The probability density of the observations.

        Raises
        ------
        ValueError
            If the shape of the last axis of `X` does not match `scales`.
        """
        if X.shape[-1] != len(scales):
            raise ValueError("Mismatch input and scale shape.")
        
        vals = []
        for i, scale in enumerate(scales):
            vals.append(stats.chi2.pdf(X[...,i]/scale, df=1)/scale)
        return np.prod(vals, axis=0)
    
    
class multivariate_normal:
    """Multivariate normal distribution with measurement errors."""
    def pdf(
        X:np.ndarray,
        cov:np.ndarray,
        err:np.ndarray=None,
        **kwargs
    ) -> np.ndarray:
        """PDF of multivariate normal distribution with measurement errors.
        The measurement errors are added to the diagonal of the covariance
        matrix.

        Parameters
        ----------
        X : (..., D) np.ndarray
            Observed values to evaluate the PDF.
        cov : (D, D) np.ndarray
            The base covariance.
        err : (D, D) np.ndarray, optional
            The measurement errors to add, by default None, treat as zeros.

        Returns
        -------
        np.ndarray
            PDF computed.
        """
        if err is not None:
            cov = cov + err
        return stats.multivariate_normal.pdf(X, cov=cov, **kwargs)
    
    def rvs(
        cov:np.ndarray, 
        size:tuple[int,...], 
        err:np.ndarray=None, 
        **kwargs
    ) -> np.ndarray:
        """Generate random samples based on the covariance and measurement
        errors specified.

        Parameters
        ----------
        cov : (D, D) np.ndarray
            The base covariance.
        size : tuple[int,...]
            Generated sample shape. Inputted to `scipy`'s `rvs` function.
        err : (D, D) np.ndarray, optional
            The measurement errors to add, by default None, treat as zeros.

        Returns
        -------
        np.ndarray
            Generated random samples of shape (size, D).
        """
        if err is not None:
            cov = cov + err
        return stats.multivariate_normal.rvs(cov=cov, size=size, **kwargs)
    
    def mle(tm:TraceModel) -> dict:
        """Calculate the updated measurement errors.

        Parameters
        ----------
        tm : TraceModel
            Model that has gone through at least one iteration so that `u` and
            `v` are available.

        Returns
        -------
        dict
            A dictionary with the new measurement errors under the key `err`.
            The value is a list of measurement error matrix at each state.
        """
        l_sqs = []
        for d in range(tm._X.shape[-1]):
            a, b = 0, 0
            for s in range(tm.S):
                s_sq = tm._dist_params[s]["cov"][d,d]
                a += np.nansum(tm._u[...,s]*(np.square(tm._X[...,d]) - s_sq))
                b += np.sum(tm._u[...,s][~np.isnan(tm._X[...,0])])
            l_sqs.append(a/b)
        L = np.diag(np.where(np.array(l_sqs) < 0, 0, l_sqs))
        param = {"err":[L.copy() for _ in range(tm.S)]}
        return param