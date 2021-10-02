import unittest
import numpy as np
from TVDenoising.scalar_denoising import denoise

class ScalarTVTest(unittest.TestCase):
    def test_denoise(self):
        np.random.seed(12345)
        n = 128
        orig = np.ones((n,n))
        noise = 0.1*np.random.randn(n,n)
        noisy = orig + noise
        rec = denoise(noisy,0.1,0.6,niter=1000,show=True)
        print(f'noisy:{np.linalg.norm(noisy)} rec:{np.linalg.norm(rec)}')