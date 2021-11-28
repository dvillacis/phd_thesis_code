import unittest
import numpy as np
from pylops import Gradient
from bilearning.Operators.Tgamma import Tgamma

class TestTgamma(unittest.TestCase):
    def test_tgamma(self):
        np.random.seed(12347)
        t = 12
        noise_level = 0.01
        sz = (t,t)
        # lu = int(sz[0]/2 - sz[0]/4)
        # br = int(lu + sz[0]/3)
        # image = np.ones(sz)
        # x = np.linspace(0, 1, t//3)
        # image[lu:br, lu:br] = np.tile(x, (t//3, 1))
        # noise = np.random.randn(sz[0], sz[1])
        # noisy = image + noise_level * noise
        # noisy = np.clip(noisy, 0.0, 1.0)
        lu = int(sz[0]/2 - sz[0]/4)
        br = int(lu + sz[0]/3)
        image = np.ones(sz)
        image[lu:br,lu:br] = 1e-5
        noisy = image + noise_level * np.random.randn(t,t)
        noisy = np.clip(noisy,0.0,1.0)
        T = Tgamma(noisy.ravel(),gamma=1e-3)
        w = np.random.randn(t*t)
        print(T)
        print(T.shape)
        print(T*w)