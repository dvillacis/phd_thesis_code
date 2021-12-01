import unittest
import warnings
import numpy as np
import matplotlib.pyplot as plt

from bilearning.TVDenoising.scalar_denoising import denoise
from bilearning.Learning.cost import l2_cost
from bilearning.Dataset.load_dataset import get_image_pair
from bilearning.Learning.reg_gradient import scalar_reg_gradient

class TestGradient(unittest.TestCase):
    def test_smooth_scalar_reg_gradient(self):
        warnings.filterwarnings('ignore')
        orig,noisy = get_image_pair('datasets/faces_train_128_10/filelist.txt',0)
        # np.random.seed(12347)
        # t = 12
        # noise_level = 0.2
        # sz = (t,t)
        # lu = int(sz[0]/2 - sz[0]/4)
        # br = int(lu + sz[0]/3)
        # orig = np.ones(sz)
        # orig[lu:br,lu:br] = 1e-5
        # noisy = orig + noise_level * np.random.randn(t,t)
        # noisy = np.clip(noisy,0.0,1.0)
        step = 1e-2
        lb = 0.01
        par = np.arange(0.01,0.2,step=step)
        grads=[]
        costs =[]
        fd_grads=[]
        rec = denoise(noisy,1.0,0.0009,niter=5000)
        c_ = l2_cost(orig,rec)
        for p in par:
            rec = denoise(noisy,1.0,p,niter=5000)
            c = l2_cost(orig,rec)
            g = scalar_reg_gradient(orig,noisy,rec,p,show=True)
            grads.append(g)
            fd_g = (c-c_)/step
            fd_grads.append(fd_g)
            print(f'p: {p:.5f}, c:{c:.7f}, g:{g}, fd_g:{fd_g}')
            costs.append(c)
            c_ = c
        fig,ax = plt.subplots()
        ax.plot(par,grads,color='red')
        ax.plot(par,fd_grads,color='green')
        ax2 = ax.twinx()
        ax2.plot(par,costs)
        ax.grid()
        plt.show()