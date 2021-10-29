import unittest
import numpy as np
from pylops import Gradient
from Operators.Tgamma import Tgamma

class TestTgamma(unittest.TestCase):
    def test_tgamma(self):
        np.random.seed(12347)
        n = 12
        u = np.random.randn(n,n)
        T = Tgamma(u.ravel(),gamma=1e-5)
        w = np.random.randn(144)
        print(T)
        print(T.shape)
        print(T*w)