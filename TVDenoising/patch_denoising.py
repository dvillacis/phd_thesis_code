import numpy as np
import pylops, pyproximal
from PIL import Image

from Dataset.load_dataset import load_ds_file
from Operators.SDL2 import SDL2
from Operators.SDL21 import SDL21

def patch_denoise(noisy,data_parameter:np.ndarray,reg_parameter:np.ndarray,niter=100,show=False):
    # if data_parameter.shape != noisy.shape or reg_parameter.shape != noisy.shape:
    #     raise ValueError('Patch parameter must be a numpy array with the same size as the input image...')
    nx,ny = noisy.shape
    K = pylops.Gradient(dims=(nx,ny),kind='forward')
    l2 = SDL2(noisy.ravel(),sigma=data_parameter.ravel())
    l21 = SDL21(ndim=2,sigma=reg_parameter.ravel())
    L = 8.0 # TODO: Estimar mejor los parametros de cp
    tau = 1.0/np.sqrt(L)
    mu = 1.0/np.sqrt(L)
    img = pyproximal.optimization.primaldual.PrimalDual(l2,l21,K,np.zeros_like(noisy.ravel()),tau,mu,niter=niter,show=show)
    return np.reshape(img,noisy.shape)

def patch_denoise_ds(dsfile,data_parameter:np.ndarray,reg_parameter:np.ndarray,niter=100,show=False):
    ds = load_ds_file(dsfile)
    reconstruction = {}
    for img in ds.keys():
        original = np.array(Image.open(img.strip()))
        original = original / np.max(original)
        noisy = np.array(Image.open(ds[img].strip()))
        noisy = noisy / np.max(noisy)
        rec = patch_denoise(noisy,data_parameter,reg_parameter,niter=niter,show=show)
        reconstruction.update({img:(original,noisy,rec)})
    return reconstruction