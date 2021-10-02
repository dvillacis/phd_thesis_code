import unittest
import numpy as np
from Learning.cost import l2_cost,l2_cost_ds, psnr_cost_ds, ssim_cost_ds
from TVDenoising.scalar_denoising import denoise_ds

class LearningTest(unittest.TestCase):
    def test_l2cost(self):
        np.random.seed(12345)
        n = 128
        orig = np.ones((n,n))
        noise = 0.1*np.random.randn(n,n)
        noisy = orig + noise
        c = l2_cost(orig,noisy)
        print(c)
    
    def test_ds_l2cost(self):
        den_ds = denoise_ds('datasets/faces_train_128_10/filelist.txt',20,1.0)
        cost = l2_cost_ds(den_ds)
        ssim = ssim_cost_ds(den_ds)
        psnr = psnr_cost_ds(den_ds)
        print(f'l2:{cost} ssim:{ssim} psnr:{psnr}')