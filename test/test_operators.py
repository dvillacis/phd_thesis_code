import unittest
import numpy as np
from Operators.operators import ActiveOp,InactiveOp

class OperatorsTest(unittest.TestCase):
    def test_activeop(self):
        np.random.seed(12345)
        n = 128
        u = np.random.rand(n,n)
        Act = ActiveOp(u)
        Inact = InactiveOp(u)
        print(Act.shape)
        print(Act.todense())
        print(Inact.todense())