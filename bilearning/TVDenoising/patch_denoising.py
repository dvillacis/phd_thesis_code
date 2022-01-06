import numpy as np
import pylops, pyproximal
from PIL import Image

from bilearning.Dataset.load_dataset import load_ds_file
from bilearning.Operators.SDL2 import SDL2
from bilearning.Operators.SDL21 import SDL21
from bilearning.Operators.patch import Patch

def patch_denoise(noisy,data_parameter:Patch,reg_parameter:Patch,niter=100,show=False):
    nx,ny = noisy.shape
    data_parameter = data_parameter.map_to_img(noisy)
    reg_parameter = reg_parameter.map_to_img(noisy)
    
    K = pylops.Gradient(dims=(nx,ny),kind='forward')
    l2 = SDL2(noisy.ravel(),sigma=data_parameter.ravel())
    l21 = SDL21(ndim=2,sigma=reg_parameter.ravel())
    L = 8.0 # TODO: Estimar mejor los parametros de cp
    tau = 1.0/np.sqrt(L)
    mu = 1.0/np.sqrt(L)
    img,_ = pyproximal.optimization.primaldual.AdaptivePrimalDual(l2,l21,K,np.zeros_like(noisy.ravel()),tau,mu,niter=niter,show=show)
    #img = pyproximal.optimization.primaldual.PrimalDual(l2,l21,K,np.zeros_like(noisy.ravel()),tau,mu,niter=niter,show=show)
    return np.reshape(img,noisy.shape)

def patch_denoise_ds(dsfile,data_parameter:Patch,reg_parameter:Patch,niter=100,show=False):
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