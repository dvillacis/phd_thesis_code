import unittest
import numpy as np
import matplotlib.pyplot as plt

from TVDenoising.scalar_denoising import denoise
from Learning.cost import l2_cost
from Dataset.load_dataset import get_image_pair
from Learning.data_gradient import smooth_scalar_data_gradient
from Learning.reg_gradient import smooth_scalar_reg_gradient

class TestGradient(unittest.TestCase):
    def test_smooth_scalar_reg_gradient(self):
        orig,noisy = get_image_pair('datasets/cameraman_128_5/filelist.txt',0)
        par = np.arange(0.0154,0.0156,step=0.1e-4)
        grads=[]
        costs =[]
        fd_grads = []
        gamma = 1e-5
        rec = denoise(noisy,1.0,0.01539,niter=5000)
        c_ = l2_cost(orig,rec)
        for p in par:
            rec = denoise(noisy,1.0,p,niter=5000)
            c = l2_cost(orig,rec)
            g = smooth_scalar_reg_gradient(orig,noisy,rec,p,show=True,gamma=gamma)
            grads.append(g)
            fd_g = (c-c_)/0.1e-4
            fd_grads.append(fd_g)
            print(f'p: {p:.5f}, c:{c:.7f}, g:{g}, fd_g:{fd_g}')
            costs.append(c)
            c_=c
        fig,ax = plt.subplots()
        ax.plot(par,grads,color='red')
        ax.plot(par,fd_grads,color='green')
        ax2 = ax.twinx()
        ax2.plot(par,costs)
        ax.grid()
        plt.show()