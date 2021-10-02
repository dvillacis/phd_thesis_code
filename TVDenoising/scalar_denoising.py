
import numpy as np
from PIL import Image
import pylops, pyproximal

from Dataset.load_dataset import load_ds_file

def denoise(noisy,data_parameter,reg_parameter,niter=100,show=False):
    nx,ny = noisy.shape
    K = pylops.Gradient(dims=(nx,ny),kind='forward')
    l2 = pyproximal.L2(b=noisy.ravel(),sigma=data_parameter)
    l21 = pyproximal.L21(ndim=2,sigma=reg_parameter)
    L = 8.0 # TODO: Estimar mejor los parametros de cp
    tau = 1.0/np.sqrt(L)
    mu = 1.0/np.sqrt(L)
    img = pyproximal.optimization.primaldual.PrimalDual(l2,l21,K,np.zeros_like(noisy.ravel()),tau,mu,niter=niter,show=show)
    return np.reshape(img,noisy.shape)

def denoise_ds(dsfile,data_parameter,reg_parameter,niter=100,show=False):
    ds = load_ds_file(dsfile)
    reconstruction = {}
    for img in ds.keys():
        original = np.array(Image.open(img.strip()))
        original = original / np.max(original)
        noisy = np.array(Image.open(ds[img].strip()))
        noisy = noisy / np.max(noisy)
        rec = denoise(noisy,data_parameter,reg_parameter,niter=niter,show=show)
        reconstruction.update({img:(original,noisy,rec)})
    return reconstruction