import unittest
import warnings
import numpy as np

from TrustRegion.nstr import nstr
from TrustRegion.step_finder import dogleg_step_finder

class TestNSTR(unittest.TestCase):
    def test_nstr(self):
        warnings.filterwarnings('ignore')
        sol = nstr(lambda x:x**2+x,lambda x:2*x+1,3.0,step_finder=dogleg_step_finder,verbose=True)
        print(sol)

        