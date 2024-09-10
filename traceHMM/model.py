""" The TraceModel class and a simulator for Markov chains.
"""

from collections import deque
from itertools import product
import numpy as np
from scipy import stats
from .cpp.update import scaled_forward, scaled_backward # type: ignore

class TraceModel:
    """Modified hidden Markov model to infer chromatin loops.
    
        Parameters
        ----------
        X : (N, T) np.ndarray
            Spatial distance of N trajectories over T time points.
        Pm : (S, S) np.ndarray
            Initial transition matrix. Entries with negative values will be 
            updated. Entries with positive values will not be updated and will 
            be the same after fitting.
        dist_params : tuple[dict, ...]
            Tuple of length S. Each element is a dictionary specifying the 
            parameters of the distribution at each state.
        dist_type : _type_, optional
            A distribution class that has a `pdf` method, by default stats.norm. 
            `dist_params` will be passed in as keyword arguments.
    """
    def __init__(
            self, 
            X:np.ndarray,
            Pm:np.ndarray,
            dist_params:tuple[dict, ...],
            dist_type=stats.norm
    ):
        (self._N, self._T), self._S = X.shape, Pm.shape[0]
        self._dist_params = dist_params
        self._dist_type = dist_type
        self._X = X
        self._convergence = []
        self._m = np.where(Pm < 0)

        # expect row sum except the fixed entries
        self._l = 1 - np.sum(np.where(Pm<0, 0, Pm), axis=1)
        initP = self._l/np.sum(Pm<0, axis=1)
        initP = np.repeat(initP[:,None], self._S, axis=1)
        # initialize transition matrix
        self._P = np.where(Pm < 0, initP, Pm)

        # initialize initial distribution
        self._mu = np.ones(self._S)/self._S

    @property
    def P(self) -> np.ndarray:
        """(S, S) np.ndarray Transition probability.
        """
        return self._P
    
    @property
    def mu(self) -> np.ndarray:
        """mu : (S) np.ndarray Initial distribution. Initialized to be uniform in each state."""
        return self._mu
    
    @property
    def N(self) -> int:
        """N : int Number of trajectories in `X`."""
        return self._N
    
    @property
    def T(self) -> int:
        """" T : int Number of time points in `X`."""
        return self._T
    
    @property
    def S(self) -> int:
        """S : int Number of states."""
        return self._S
    
    @property
    def convergence(self) -> np.ndarray:
        """convergence : list
            The mean absolute difference between the updated transition matrix 
            and the transition matrix from the last iteration.
        """
        return self._convergence

    def density(
            self, 
            state:int|np.ndarray, 
            x:int|np.ndarray
    ) -> int|np.ndarray:
        """Calculates the probability of given observation(s) at state(s).

        Parameters
        ----------
        state : int | np.ndarray
            A single state or an array of states to evaluate x.
        x : int | np.ndarray
            Observations. Either a number or an array.

        Returns
        -------
        int | np.ndarray
            Probability of observing x at state(s). Will broadcast.
        """
        if np.issubdtype(type(state), np.integer):
            args = self._dist_params[state]
            vals = self._dist_type.pdf(x, **args)
            if isinstance(x, float):
                if np.isnan(x):
                    return 1
                return vals
        elif isinstance(x, float):
            vals = []
            for s in state:
                args = self._dist_params[s]
                vals.append(self._dist_type.pdf(x, **args))
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
        v = np.zeros((self._N, self._T, self._S, self._S))
        states = np.arange(self._S)
        for s1, s2 in product(states, states):
            for t in range(1, self._T):
                den = self.density(s2, X[:,t])
                raw = den*self._b[:,t,s2]*self._f[:,t-1,s1]*self._P[s1,s2]
                frac = np.sum(self._f[:,t,:]*self._b[:,t,:], axis=1)*self._alpha[:,t]
                v[:,t,s1,s2] = raw/frac
        self._v = v

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
            self._backward_(self._X)

            self._calculate_u_()
            self._calculate_v_(self._X)

            self._update_params_()
            if self._convergence[-1] < cutoff:
                print(f"Converged at iteration {len(self._convergence)}")
                break

    def _viterbi_(self, x:np.ndarray) -> np.ndarray:
        v = np.zeros((len(x), self._S))
        varg = np.zeros((len(x), self._S), dtype="int64")
        states = np.arange(self._S)

        v[0] = np.log(self.density(states, x[0])) + np.log(self._mu)
        for t in range(1, len(x)):
            den = self.density(states, x[t])
            logP = np.where(self._P==0, -np.inf, np.log(np.where(self._P==0, 1, self._P)))
            val = v[t-1][:,None] + logP + np.log(den[None,:])
            v[t] = np.max(val, axis=0)
            varg[t-1] = np.argmax(val, axis=0)

        decoded_states = deque([np.argmax(v[-1])])
        for t in range(len(x)-2, -1, -1):
            decoded_states.appendleft(varg[t, decoded_states[0]])
        return np.array(decoded_states)
    
    def decode(self, X:np.ndarray=None) -> np.ndarray:
        """Predict the looping status of the input data using the fitted 
        model. Has to be run after `fit`.

        Parameters
        ----------
        X : (N', T') np.ndarray, optional
            New chromatin trace to predict looping status, by default None. If
            None is passed, will return the predicted loop status of the input
            used to train the model.

        Returns
        -------
        np.ndarray
            Same shape as `X`. Contains the prediction result.
        """
        if X is None:
            X = self._X
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
            dist_type=stats.norm
    ):
        self._P = P
        self._mu = mu
        self._dist_params = dist_params
        self._dist_type = dist_type
        
    @property
    def P(self) -> np.ndarray:
        """P : (S, S) np.ndarray Transition probability."""
        return self._P
    
    @property
    def mu(self) -> np.ndarray:
        """mu : (S) np.ndarray Initial distribution. Initialized to be uniform in each state."""
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
        tuple[(T) np.ndarray, (T) np.ndarray]
            The first element is the true loop status of the trace. The second
            element is the spatial distance generated at each time point.
        """
        H = [np.random.choice(len(self._mu), p=self._mu)]
        for t in range(1, T):
            H.append(np.random.choice(len(self._mu), p=self._P[H[-1]]))
        H = np.array(H)
        
        X = np.zeros_like(H)
        for s in range(len(self._mu)):
            args = self._dist_params[s]
            x = self._dist_type.rvs(**args, size=T)
            X = np.where(H==s, x, X)
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
        result =  np.stack([
            self.simulate_single_trace(T) for _ in range(N)
        ])
        H = result[:,0,:].astype("int64")
        X = result[:,1,:]
        return H, X
    
    @staticmethod
    def mask_spatial_distance(
        self, X:np.ndarray, p_obs:float
    ) -> np.ndarray:
        """Mask spatial distance uniformly at random to simulate the missing
        data problem observed in real imaging data.

        Parameters
        ----------
        X : (..., T) np.ndarray
            Input spatial distance to be masked.
        p_obs : float
            The observed probability.

        Returns
        -------
        (..., T) np.ndarray
            Same shape as `X` with missing observations filled by NaN.
        """
        mask = np.random.choice(2, size=X.shape, p=[1 - p_obs, p_obs])
        masked_X = np.where(mask==0, np.nan, X)
        return masked_X
    
    def mask_by_markov_chain(
        self, X:np.ndarray, p_obs:float, a:float=0.8
    ) -> np.ndarray:
        """Mask spatial distance by a two-state Markov chain. This is inspired 
        by the fact that missing observations are typically continuous in real
        imaging data.

        Parameters
        ----------
        X : (..., T) np.ndarray
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
        mc = TraceSimulator(P, mu, self._dist_params)
        mask = mc.simulate_multiple_traces(X.shape[1], X.shape[0])[0]
        return np.where(mask==0, np.nan, X)
