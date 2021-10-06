import unittest
from PIL import Image
import numpy as np
from TVDenoising.patch_denoising import denoise, denoise_ds

class PatchTVTest(unittest.TestCase):
    def test_denoise(self):
        np.random.seed(12345)
        n = 128
        orig = np.ones((n,n))
        noise = 0.1*np.random.randn(n,n)
        noisy = orig + noise
        data_parameter = 60.0 * np.ones((n,n))
        reg_parameter = 0.1 * np.ones((n,n))
        rec = denoise(noisy,data_parameter,reg_parameter,niter=1000,show=True)
        print(f'noisy:{np.linalg.norm(noisy)} rec:{np.linalg.norm(rec)}')

    def test_ds_denoise(self):
        n = 128
        data_parameter = 60.0 * np.ones((n,n))
        reg_parameter = 0.1 * np.ones((n,n))
        rec = denoise_ds('datasets/faces_train_128_10/filelist.txt',data_parameter,reg_parameter,show=True)
        print(len(rec))
