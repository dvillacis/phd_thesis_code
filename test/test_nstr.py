import unittest
import warnings
import numpy as np

from TrustRegion.nsdogbox import nsdogbox
from TrustRegion.step_finder import dogleg_step_finder,cauchy_point_step_finder

class TestNSTR(unittest.TestCase):
    def test_nstr(self):
        warnings.filterwarnings('ignore')
        A = np.array([[1.0,0.0],[0.0,0.5]])
        v = np.array([1.0,1.0])
        fun = lambda x:0.5*np.dot(x,np.dot(A,x))+np.dot(v,x)
        grad = lambda x: np.dot(A,x)+v
        x0 = np.array([300.0,300.0])
        lb = np.array([0.0,0.0])
        ub = np.array([1000.0,1000.0])
        sol = nsdogbox(fun,grad,x0,lb,ub,verbose=2)
        print(sol)

        