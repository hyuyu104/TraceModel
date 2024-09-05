import unittest
from collections import deque
import numpy as np
from scipy import stats
from . import model


def scaled_forward(X, tm):
    (N, T), S = X.shape, len(tm.mu)
    f = np.zeros((N, T, S))
    for n in range(N):
        for t in range(T):
            for s in range(S):
                if t == 0:
                    f[n,t,s] = tm.mu[s]*tm.density(s, X[n,t])
                else:
                    den = tm.density(s, X[n,t])
                    f[n,t,s] = den*sum([f[n,t-1,sp]*tm.P[sp,s] for sp in range(S)])
            f[n,t,:] = f[n,t,:]/np.sum(f[n,t,:])
    return f


def scaled_backward(X, tm):
    (N, T), S = X.shape, len(tm.mu)
    b = np.zeros((N, T, S))
    for n in range(N):
        for t in range(T-1, -1, -1):
            for s in range(S):
                if t == T-1:
                    b[n,t,s] = 1
                else:
                    b[n,t,s] = sum([
                        b[n,t+1,sp]*tm.P[s,sp]*tm.density(sp,X[n,t+1])
                        for sp in range(S)
                    ])
            b[n,t,:] = b[n,t,:]/sum(b[n,t,:])
    return b


def log_viterbi(x, tm):
    T, S = len(x), len(tm.mu)
    v, varg = np.zeros((T, S)), np.zeros((T, S), dtype="int64")
    for t in range(T):
        for s in range(S):
            if t == 0:
                v[t,s] = np.log(tm.mu[s]) + np.log(tm.density(s,x[t]))
            else:
                vs = np.array([
                    v[t-1,h] + np.log(tm.P[h,s]) + np.log(tm.density(s,x[t]))
                    if tm.P[h,s] > 0 else -np.inf for h in range(S)
                ])
                v[t,s] = max(vs)
                varg[t-1,s] = np.argmax(vs)
    varg[t] = np.argmax(v[t])
    decoded_states = deque([varg[-1,0]])
    for t in range(len(x)-2, -1, -1):
        decoded_states.appendleft(varg[t, decoded_states[0]])
    return decoded_states



class TestFittingTwoStates(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [10, 12, 13, -4, -7, -5, 8],
            [-10, -8, 10, 14, 11, 6, 8]
        ], dtype="float")
        self.P = np.array([
            [-1, -1],
            [-1, -1]
        ])
        self.dist_params = (
            {"loc":10, "scale":3},
            {"loc":-8, "scale":3},
        )
        self.tm = model.TraceModel(self.X, self.P, self.dist_params)

    def test_density(self):
        true_val = stats.norm.pdf(
            0.5, 
            loc=self.dist_params[0]["loc"], 
            scale=self.dist_params[0]["scale"]
        )
        val = self.tm.density(0, 0.5)
        self.assertAlmostEqual(val, true_val)

        true_val2 = stats.norm.pdf(
            0.5, 
            loc=self.dist_params[1]["loc"], 
            scale=self.dist_params[1]["scale"]
        )
        true_val = np.array([true_val, true_val2, true_val])
        val = self.tm.density(np.array([0, 1, 0]), 0.5)
        self.assertAlmostEqual(np.sum(np.abs(val - true_val)), 0)

        val = self.tm.density(np.array([0, 1, 0]), np.array([0.1, 0.8]))
        self.assertEqual(val.shape, (3, 2))
        val = self.tm.density(np.array([0, 1]), self.X[0])
        self.assertEqual(val.shape, (2, 7))

        self.assertEqual(self.tm.density(1, np.nan), 1)
        val = self.tm.density(np.array([0, 1, 0]), np.array([np.nan, 0.8]))
        self.assertTrue(np.all(val[:,0] == 1))

    def test_forward_no_missing(self):
        self.tm._forward_(self.X)
        val = self.tm.f
        self.assertEqual(val.shape, (2, 7, 2))

        f = scaled_forward(self.X, self.tm)
        err = np.mean(np.abs(f - val))
        self.assertAlmostEqual(err, 0)

    def test_forward_missing1(self):
        random_mask = np.random.choice(2, self.X.shape)
        random_mask[[0,0]] = 0
        random_mask[[1,-1]] = 0
        X = np.where(random_mask==0, np.nan, self.X)
        self.tm._forward_(X)
        val = self.tm.f
        f = scaled_forward(X, self.tm)
        err = np.mean(np.abs(f - val))
        self.assertAlmostEqual(err, 0)

    def test_forward_missing2(self):
        random_mask = np.random.choice(2, self.X.shape)
        random_mask[[0,0]] = 1
        random_mask[[1,-1]] = 0
        X = np.where(random_mask==0, np.nan, self.X)
        self.tm._forward_(X)
        val = self.tm.f
        f = scaled_forward(X, self.tm)
        err = np.mean(np.abs(f - val))
        self.assertAlmostEqual(err, 0)

    def test_forward_missing3(self):
        random_mask = np.random.choice(2, self.X.shape)
        random_mask[[0,0]] = 0
        random_mask[[1,-1]] = 1
        X = np.where(random_mask==0, np.nan, self.X)
        self.tm._forward_(X)
        val = self.tm.f
        f = scaled_forward(X, self.tm)
        err = np.mean(np.abs(f - val))
        self.assertAlmostEqual(err, 0)

    def test_backward_no_missing(self):
        self.tm._backward_(self.X)
        val = self.tm.b
        self.assertEqual(val.shape, (2, 7, 2))

        b = scaled_backward(self.X, self.tm)
        err = np.mean(np.abs(b - val))
        self.assertAlmostEqual(err, 0)

    def test_backward_missing1(self):
        random_mask = np.random.choice(2, self.X.shape)
        random_mask[[0,0]] = 0
        random_mask[[1,-1]] = 0
        X = np.where(random_mask==0, np.nan, self.X)
        self.tm._backward_(X)
        val = self.tm.b
        b = scaled_backward(X, self.tm)
        err = np.mean(np.abs(b - val))
        self.assertAlmostEqual(err, 0)

    def test_backward_missing2(self):
        random_mask = np.random.choice(2, self.X.shape)
        random_mask[[0,0]] = 1
        random_mask[[1,-1]] = 0
        X = np.where(random_mask==0, np.nan, self.X)
        self.tm._backward_(X)
        val = self.tm.b
        b = scaled_backward(X, self.tm)
        err = np.mean(np.abs(b - val))
        self.assertAlmostEqual(err, 0)

    def test_backward_missing3(self):
        random_mask = np.random.choice(2, self.X.shape)
        random_mask[[0,0]] = 0
        random_mask[[1,-1]] = 1
        X = np.where(random_mask==0, np.nan, self.X)
        self.tm._backward_(X)
        val = self.tm.b
        b = scaled_backward(X, self.tm)
        err = np.mean(np.abs(b - val))
        self.assertAlmostEqual(err, 0)


class TestFittingThreeStates(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [10, 12, 13, -4, -7, -5, 8],
            [-10, -8, 10, 14, 11, 6, 8]
        ], dtype="float")
        self.P = np.array([
            [-1, -1, 0],
            [-1, -1, -1],
            [0, -1, -1]
        ])
        self.dist_params = (
            {"loc":10, "scale":3},
            {"loc":0, "scale":3},
            {"loc":-8, "scale":3},
        )
        self.tm = model.TraceModel(self.X, self.P, self.dist_params)

    def test_forward_no_missing(self):
        self.tm._forward_(self.X)
        val = self.tm.f
        self.assertEqual(val.shape, (2, 7, 3))

        f = scaled_forward(self.X, self.tm)
        err = np.mean(np.abs(f - val))
        self.assertAlmostEqual(err, 0)

    def test_forward_missing1(self):
        random_mask = np.random.choice(2, self.X.shape)
        random_mask[[0,0]] = 0
        random_mask[[1,-1]] = 0
        X = np.where(random_mask==0, np.nan, self.X)
        self.tm._forward_(X)
        val = self.tm.f
        f = scaled_forward(X, self.tm)
        err = np.mean(np.abs(f - val))
        self.assertAlmostEqual(err, 0)

    def test_forward_missing2(self):
        random_mask = np.random.choice(2, self.X.shape)
        random_mask[[0,0]] = 1
        random_mask[[1,-1]] = 0
        X = np.where(random_mask==0, np.nan, self.X)
        self.tm._forward_(X)
        val = self.tm.f
        f = scaled_forward(X, self.tm)
        err = np.mean(np.abs(f - val))
        self.assertAlmostEqual(err, 0)

    def test_forward_missing3(self):
        random_mask = np.random.choice(2, self.X.shape)
        random_mask[[0,0]] = 0
        random_mask[[1,-1]] = 1
        X = np.where(random_mask==0, np.nan, self.X)
        self.tm._forward_(X)
        val = self.tm.f
        f = scaled_forward(X, self.tm)
        err = np.mean(np.abs(f - val))
        self.assertAlmostEqual(err, 0)

    def test_backward_no_missing(self):
        self.tm._backward_(self.X)
        val = self.tm.b
        self.assertEqual(val.shape, (2, 7, 3))

        b = scaled_backward(self.X, self.tm)
        err = np.mean(np.abs(b - val))
        self.assertAlmostEqual(err, 0)

    def test_backward_missing1(self):
        random_mask = np.random.choice(2, self.X.shape)
        random_mask[[0,0]] = 0
        random_mask[[1,-1]] = 0
        X = np.where(random_mask==0, np.nan, self.X)
        self.tm._backward_(X)
        val = self.tm.b
        b = scaled_backward(X, self.tm)
        err = np.mean(np.abs(b - val))
        self.assertAlmostEqual(err, 0)

    def test_backward_missing2(self):
        random_mask = np.random.choice(2, self.X.shape)
        random_mask[[0,0]] = 1
        random_mask[[1,-1]] = 0
        X = np.where(random_mask==0, np.nan, self.X)
        self.tm._backward_(X)
        val = self.tm.b
        b = scaled_backward(X, self.tm)
        err = np.mean(np.abs(b - val))
        self.assertAlmostEqual(err, 0)

    def test_backward_missing3(self):
        random_mask = np.random.choice(2, self.X.shape)
        random_mask[[0,0]] = 0
        random_mask[[1,-1]] = 1
        X = np.where(random_mask==0, np.nan, self.X)
        self.tm._backward_(X)
        val = self.tm.b
        b = scaled_backward(X, self.tm)
        err = np.mean(np.abs(b - val))
        self.assertAlmostEqual(err, 0)


class TestDecoding(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [10, 12, 13, -4, -7, -5, -8],
            [-10, -8, 10, 14, 11, 6, 8]
        ], dtype="float")
        self.P = np.array([
            [-1, -1, 0],
            [-1, -1, -1],
            [0, -1, -1]
        ])
        self.dist_params = (
            {"loc":10, "scale":3},
            {"loc":0, "scale":3},
            {"loc":-8, "scale":3},
        )
        self.tm = model.TraceModel(self.X, self.P, self.dist_params)
        self.tm.fit()

    def test_viterbi(self):
        val = self.tm.decode()
        print(val)
        true_val = np.stack([log_viterbi(x, self.tm) for x in self.X])
        print(true_val)
        self.assertAlmostEqual(np.mean(np.abs(val - true_val)), 0)

if __name__ == "__main__":
    unittest.main()