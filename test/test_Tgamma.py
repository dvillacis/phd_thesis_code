import unittest
import numpy as np
from pylops import Gradient
from Operators.Tgamma import Tgamma

class TestTgamma(unittest.TestCase):
    def test_tgamma(self):
        np.random.seed(12345)
        n = 12
        u = np.random.rand(n,n)
        K = Gradient(dims=(n,n),kind='forward')
        Ku = K*u.ravel()
        print(Ku.shape)
        w = np.ones(len(Ku))
        T = Tgamma(Ku,gamma=1e-5)
        print(T)
        print(T*K)
        print(T.shape)
        print(T*w)