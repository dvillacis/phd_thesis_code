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
        par = np.arange(0.01,0.03,step=0.0005)
        grads=[]
        costs =[]
        for p in par:
            rec = denoise(noisy,1.0,p,niter=1000)
            c = l2_cost(orig,rec)
            g = smooth_scalar_reg_gradient(orig,noisy,rec,p,show=True)
            grads.append(g)
            print(f'p: {p:.4f}, c:{c:.7f}, g:{g}')
            costs.append(c)
        fig,ax = plt.subplots()
        ax.plot(par,grads,color='red')
        ax2 = ax.twinx()
        ax2.plot(par,costs)
        ax.grid()
        plt.show()