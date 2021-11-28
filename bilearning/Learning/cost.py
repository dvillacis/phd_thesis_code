import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def l2_cost(original,reconstruction):
    return 0.5*np.linalg.norm(original.ravel()-reconstruction.ravel())**2

def l2_cost_ds(ds_denoised):
    cost = 0
    for img in ds_denoised.keys():
        cost += l2_cost(ds_denoised[img][0],ds_denoised[img][2])
    return cost/len(ds_denoised)

def ssim_cost_ds(ds_denoised):
    cost = 0
    for img in ds_denoised.keys():
        cost += ssim(ds_denoised[img][0],ds_denoised[img][2])
    return cost/len(ds_denoised)

def psnr_cost_ds(ds_denoised):
    cost = 0
    for img in ds_denoised.keys():
        cost += psnr(ds_denoised[img][0],ds_denoised[img][2])
    return cost/len(ds_denoised)
    