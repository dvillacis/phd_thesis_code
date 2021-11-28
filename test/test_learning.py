import unittest
import numpy as np
from bilearning.Learning.cost import l2_cost,l2_cost_ds, psnr_cost_ds, ssim_cost_ds
from bilearning.TVDenoising.scalar_denoising import denoise, denoise_ds
from bilearning.Learning.data_gradient import scalar_data_adjoint, scalar_data_gradient
from bilearning.Learning.reg_gradient import scalar_reg_gradient
from bilearning.Dataset.load_dataset import get_image_pair
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

    # def test_scalar_data_gradient(self):
    #     orig,noisy = get_image_pair('datasets/cameraman_128_5/filelist.txt',0)
    #     par = np.arange(50.0,500.0,4.0)
    #     grads=[]
    #     costs =[]
    #     for p in par:
    #         rec = denoise(noisy,p,1.0,niter=1000)
    #         c = l2_cost(orig,rec)
    #         g = scalar_data_gradient(orig,noisy,rec,p)
    #         grads.append(g)
    #         costs.append(c)
    #     fig,ax = plt.subplots()
    #     ax.plot(par,grads,color='red')
    #     ax2 = ax.twinx()
    #     ax2.plot(par,costs)
    #     ax.grid()
    #     plt.show()

    def test_scalar_reg_gradient(self):
        orig,noisy = get_image_pair('datasets/cameraman_128_5/filelist.txt',0)
        par = np.arange(0.001,0.1,0.001)
        grads=[]
        costs =[]
        for p in par:
            rec = denoise(noisy,1.0,p,niter=1000)
            c = l2_cost(orig,rec)
            g = scalar_reg_gradient(orig,noisy,rec,p)
            grads.append(g)
            #print(g)
            costs.append(c)
        fig,ax = plt.subplots()
        ax.plot(par,grads,color='red')
        ax2 = ax.twinx()
        ax2.plot(par,costs)
        ax.grid()
        plt.show()