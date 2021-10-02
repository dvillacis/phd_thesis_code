
import numpy as np
from numpy.core.fromnumeric import ndim
import pylops, pyproximal

def denoise(noisy,data_parameter,reg_parameter,niter=100,show=False):
    nx,ny = noisy.shape
    K = pylops.Gradient(dims=(nx,ny),kind='forward')
    l2 = pyproximal.L2(b=noisy.ravel(),sigma=data_parameter)
    l21 = pyproximal.L21(ndim=2,sigma=reg_parameter)
    L = 8.0
    tau = 1.0/np.sqrt(L)
    mu = 1.0/np.sqrt(L)
    img = pyproximal.optimization.primaldual.PrimalDual(l2,l21,K,noisy.ravel(),tau,mu,niter=niter,show=show)
    return np.reshape(img,noisy.shape)