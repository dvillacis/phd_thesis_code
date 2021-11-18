import unittest
import warnings
import numpy as np

from TrustRegion.nstr import nstr
from TrustRegion.step_finder import cauchy_point_step_finder

class TestNSTR(unittest.TestCase):
    def test_nstr(self):
        warnings.filterwarnings('ignore')
        sol = nstr(lambda x:np.dot(x,x)+5*x,lambda x:2*x+5,np.array([3.0,3.0]),step_finder=cauchy_point_step_finder,verbose=True)
        print(sol)

        