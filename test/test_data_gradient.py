import unittest
import numpy as np
import matplotlib.pyplot as plt

from TVDenoising.scalar_denoising import denoise
from Learning.cost import l2_cost
from Dataset.load_dataset import get_image_pair
from Learning.data_gradient import scalar_data_gradient

class TestGradient(unittest.TestCase):
    def test_smooth_scalar_data_gradient(self):
        orig,noisy = get_image_pair('datasets/cameraman_128_5/filelist.txt',0)
        par = np.arange(50.0,250.0,step=1.0)
        grads=[]
        costs =[]
        for p in par:
            rec = denoise(noisy,p,1.0,niter=1000)
            c = l2_cost(orig,rec)
            g = scalar_data_gradient(orig,noisy,rec,p,show=True)
            grads.append(g)
            print(f'p: {p}, c:{c}, g:{g}')
            costs.append(c)
        fig,ax = plt.subplots()
        ax.plot(par,grads,color='red')
        ax2 = ax.twinx()
        ax2.plot(par,costs)
        ax.grid()
        plt.show()