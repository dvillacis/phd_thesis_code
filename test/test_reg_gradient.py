import unittest
import warnings
import numpy as np
import matplotlib.pyplot as plt

from TVDenoising.scalar_denoising import denoise
from Learning.cost import l2_cost
from Dataset.load_dataset import get_image_pair
from Learning.reg_gradient import scalar_reg_gradient

class TestGradient(unittest.TestCase):
    def test_smooth_scalar_reg_gradient(self):
        warnings.filterwarnings('ignore')
        orig,noisy = get_image_pair('datasets/cameraman_128_5/filelist.txt',0)
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
        
        par = np.arange(0.0152,0.0157,step=0.1e-4)
        grads=[]
        costs =[]
        for p in par:
            rec = denoise(noisy,1.0,p,niter=10000)
            c = l2_cost(orig,rec)
            g = scalar_reg_gradient(orig,noisy,rec,p,show=True)
            grads.append(g)
            print(f'p: {p:.4f}, c:{c:.7f}, g:{g}')
            costs.append(c)
        fig,ax = plt.subplots()
        ax.plot(par,grads,color='red')
        ax2 = ax.twinx()
        ax2.plot(par,costs)
        ax.grid()
        plt.show()