import numpy as np
import pylops, pyproximal
from PIL import Image

from bilearning.Dataset.load_dataset import load_ds_file, get_image_pair_by_key
from bilearning.Operators.SDGP import SDGP
from bilearning.Operators.patch import Patch

def gp_patch_denoise(noisy,data_parameter_gausian:Patch,data_parameter_poisson:Patch,niter=100,show=False):
    nx,ny = noisy.shape
    data_parameter_gausian = data_parameter_gausian.map_to_img(noisy)
    data_parameter_poisson = data_parameter_poisson.map_to_img(noisy)
    
    K = pylops.Gradient(dims=(nx,ny),kind='forward')
    l2_poisson = SDGP(noisy.ravel(),sigma1=data_parameter_gausian.ravel(),sigma2=data_parameter_poisson.ravel())
    l21 = pyproximal.L21(ndim=2,sigma=1.)
    L = 8.0 # TODO: Estimar mejor los parametros de cp
    tau = 0.01
    mu = 1/tau/L
    img = pyproximal.optimization.primaldual.PrimalDual(l2_poisson,l21,K,np.zeros_like(noisy.ravel()),tau,mu,niter=niter,show=show)
    img = np.reshape(img, noisy.shape)
    return img


def gp_patch_denoise_ds(dsfile, data_parameter_gausian: Patch, data_parameter_poisson: Patch, niter=100, show=False):
    ds = load_ds_file(dsfile)
    reconstruction = {}
    for img in ds.keys():
        original,noisy = get_image_pair_by_key(dsfile,img)
        rec = gp_patch_denoise(noisy,data_parameter_gausian,data_parameter_poisson,niter=niter,show=show)
        reconstruction.update({img:(original,noisy,rec)})
    return reconstruction