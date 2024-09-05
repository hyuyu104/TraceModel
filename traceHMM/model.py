from typing import Callable
from collections import deque
from itertools import product
import numpy as np
from scipy import stats

class TraceModel:

    def __init__(
            self, 
            X:np.array,
            P:np.array,
            dist_params:tuple,
            dist_type=stats.norm
    ):
        (self.N, self.T), self.S = X.shape, P.shape[0]
        self.dist_params = dist_params
        self.dist_type = dist_type
        self.X = X
        self._convergence = []

        # initialize transition matrix
        row_init = 1/np.sum(P<0, axis=1)
        mat_init = np.repeat(row_init, self.S).reshape(P.shape)
        self.P = np.where(P<0, mat_init, P)

        # initialize initial distribution
        self.mu = np.ones(self.S)/self.S

    def density(self, state, x):
        if np.issubdtype(type(state), np.integer):
            args = self.dist_params[state]
            vals = self.dist_type.pdf(x, **args)
            if isinstance(x, float):
                return 1 if np.isnan(x) else vals
        elif isinstance(x, float):
            vals = []
            for s in state:
                args = self.dist_params[s]
                vals.append(self.dist_type.pdf(x, **args))
        else:
            vals = []
            for s in state:
                args = self.dist_params[s]
                val = []
                for b in x:
                    val.append(self.dist_type.pdf(b, **args))
                vals.append(val)
        return np.where(np.isnan(vals), 1, vals)
    
    def _forward_(self, X):
        f = np.zeros((self.N, self.T, self.S))
        alpha = np.zeros((self.N, self.T))
        states = np.arange(self.S)
        for t in range(self.T):
            den = self.density(states, X[:,t])
            raw = self.mu*den.T if t==0 else (f[:,t-1,:]@self.P)*den.T
            alpha[:,t] = np.sum(raw, axis=1)
            f[:,t,:] = raw/alpha[:,[t]]
        self.alpha = alpha
        self.f = f

    def _backward_(self, X):
        b = np.zeros((self.N, self.T, self.S))
        beta = np.zeros((self.N, self.T))
        states = np.arange(self.S)
        for t in range(self.T-1, -1, -1):
            if t == self.T-1:
                raw = np.ones((self.N, self.S))
            else:
                den = self.density(states, X[:,t+1])
                raw = (self.P@(b[:,t+1,:].T*den)).T
            beta[:,t] = np.sum(raw, axis=1)
            b[:,t,:] = raw/beta[:,[t]]
        self.beta = beta
        self.b = b

    def _calculate_u_(self):
        u_raw = self.f*self.b
        u_sum = np.sum(u_raw, axis=2)
        u = np.stack([a/b[:,None] for a, b in zip(u_raw, u_sum)])
        self.u = u

    def _calculate_v_(self, X):
        v = np.zeros((self.N, self.T, self.S, self.S))
        states = np.arange(self.S)
        for s1, s2 in product(states, states):
            for t in range(1, self.T):
                den = self.density(s2, X[:,t])
                raw = den*self.b[:,t,s2]*self.f[:,t-1,s1]*self.P[s1,s2]
                frac = np.sum(self.f[:,t,:]*self.b[:,t,:], axis=1)*self.alpha[:,t]
                v[:,t,s1,s2] = raw/frac
        self.v = v

    def _update_params_(self):
        self.mu = np.mean(self.u[:,0,:], axis=0)

        P_hat = np.zeros_like(self.P)
        states = np.arange(self.S)
        for s1, s2 in product(states, states):
            P_hat[s1,s2] = np.sum(self.v[:,1:,s1,s2])/np.sum(self.v[:,1:,s1,:])
        self._convergence.append(np.mean(np.abs(P_hat - self.P)))
        self.P = P_hat

    def fit(self, max_iter=100):
        for _ in range(max_iter):
            self._forward_(self.X)
            self._backward_(self.X)

            self._calculate_u_()
            self._calculate_v_(self.X)

            self._update_params_()
            if self._convergence[-1] < 1e-4:
                print(f"Converged at iteration {len(self._convergence)}")
                break

    def _viterbi_(self, x):
        v = np.zeros((len(x), self.S))
        varg = np.zeros((len(x), self.S), dtype="int64")
        states = np.arange(self.S)
        v[0] = np.log(self.density(states, x[0])) + np.log(self.mu)
        for t in range(1, self.T):
            den = self.density(states, x[t])
            logP = np.where(self.P == 0, -np.inf, np.log(np.where(self.P == 0, 1, self.P)))
            val = v[t-1][:,None] + logP + np.log(den[None,:])
            v[t] = np.max(val, axis=0)
            varg[t-1] = np.argmax(val, axis=0)

        decoded_states = deque([np.argmax(v[t])])
        for t in range(len(x)-2, -1, -1):
            decoded_states.appendleft(varg[t, decoded_states[0]])
        return np.array(decoded_states)
    
    def decode(self, X=None):
        if X is None:
            return np.stack([self._viterbi_(x) for x in self.X])
        return np.stack([self._viterbi_(x) for x in X])