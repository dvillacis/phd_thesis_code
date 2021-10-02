import numpy as np

def l2_cost(original,reconstruction):
    return 0.5*np.linalg.norm(original.ravel()-reconstruction.ravel())**2

def l2_cost_ds(ds_denoised):
    cost = 0
    for img in ds_denoised.keys():
        cost += l2_cost(ds_denoised[img][0],ds_denoised[img][2])
    return cost
    