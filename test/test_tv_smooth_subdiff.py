import unittest
import numpy as np
from pylops import Gradient
from bilearning.Operators.norms import tv_smooth_subdiff

class TestTVSubdiff(unittest.TestCase):
    def test_tvsubdiff(self):
        np.random.seed(12347)
        t = 12
        noise_level = 0.01
        sz = (t,t)
        lu = int(sz[0]/2 - sz[0]/4)
        br = int(lu + sz[0]/3)
        image = np.ones(sz)
        image[lu:br,lu:br] = 1e-5
        noisy = image + noise_level * np.random.randn(t,t)
        noisy = np.clip(noisy,0.0,1.0)
        hu = tv_smooth_subdiff(noisy.ravel())