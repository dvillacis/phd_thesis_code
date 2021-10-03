import unittest
import numpy as np
from Learning.cost import l2_cost,l2_cost_ds, psnr_cost_ds, ssim_cost_ds
from TVDenoising.scalar_denoising import denoise, denoise_ds
from Learning.data_gradient import scalar_data_adjoint, scalar_data_gradient
import matplotlib.pyplot as plt

class LearningTest(unittest.TestCase):
    def test_l2cost(self):
        np.random.seed(12345)
        n = 128
        orig = np.ones((n,n))
        noise = 0.1*np.random.randn(n,n)
        noisy = orig + noise
        c = l2_cost(orig,noisy)
        print(c)
    
    def test_ds_cost(self):
        den_ds = denoise_ds('datasets/faces_train_128_10/filelist.txt',30,1.0)
        cost = l2_cost_ds(den_ds)
        ssim = ssim_cost_ds(den_ds)
        psnr = psnr_cost_ds(den_ds)
        print(f'l2:{cost} ssim:{ssim} psnr:{psnr}')

    def test_scalar_data_adjoint(self):
        np.random.seed(12345)
        n = 10
        orig = np.ones((n,n))
        noise = 0.1*np.random.randn(n,n)
        noisy = orig + noise
        rec = denoise(noise,12.0,1.0)
        p = scalar_data_adjoint(orig,rec,12.0,show=True)

    def test_scalar_gradient(self):
        np.random.seed(12345)
        n = 128
        orig = np.ones((n,n))
        noise = 0.1*np.random.randn(n,n)
        noisy = orig + noise
        par = np.arange(5.0,100.0,1.0)
        grads=[]
        costs =[]
        for p in par:
            rec = denoise(noise,p,1.0,niter=1000)
            c = l2_cost(orig,rec)
            g = scalar_data_gradient(orig,noisy,rec,p)
            grads.append(g)
            costs.append(c)
        # print(f'\ngrad: {grads}')
        plt.plot(par,grads)
        plt.plot(par,costs)
        plt.show()