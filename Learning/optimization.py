import numpy as np
import scipy
from Learning.cost import l2_cost_ds
from TVDenoising.scalar_denoising import denoise_ds
from TVDenoising.patch_denoising import patch_denoise_ds
from Learning.data_gradient import scalar_data_gradient_ds, patch_data_gradient_ds

#################################
# SCALAR DATA PARAMETER LEARNING
#################################

def data_cost_fn_scalar(dsfile,data_parameter):
    den_ds = denoise_ds(dsfile,data_parameter=data_parameter,reg_parameter=1.0)
    return l2_cost_ds(den_ds)

def data_gradient_fn_scalar(dsfile,data_parameter):
    den_ds = denoise_ds(dsfile,data_parameter=data_parameter,reg_parameter=1.0)
    return scalar_data_gradient_ds(den_ds,data_parameter)

def find_optimal_data_scalar(dsfile,initial_data_parameter,show=False):
    iprint = -1
    if show == True:
        iprint = 100
    optimal = scipy.optimize.minimize(fun=lambda x: data_cost_fn_scalar(dsfile,x),jac=lambda x:data_gradient_fn_scalar(dsfile,x),x0=initial_data_parameter,method='L-BFGS-B',options={'iprint':iprint})
    if show == True:
        print(optimal)
    optimal_ds = denoise_ds(dsfile,data_parameter=optimal.x,reg_parameter=1.0)
    return optimal,optimal_ds

#################################
# PATCH DATA PARAMETER LEARNING
#################################

def data_cost_fn_patch(dsfile,data_parameter:np.ndarray):
    den_ds = patch_denoise_ds(dsfile,data_parameter=data_parameter,reg_parameter=np.ones(data_parameter.shape),niter=1000)
    return l2_cost_ds(den_ds)

def data_gradient_fn_patch(dsfile,data_parameter:np.ndarray):
    den_ds = patch_denoise_ds(dsfile,data_parameter=data_parameter,reg_parameter=np.ones(data_parameter.shape),niter=1000)
    grad = patch_data_gradient_ds(den_ds,data_parameter)
    return grad

def find_optimal_data_patch(dsfile,initial_data_parameter,show=False):
    initial_data_parameter = initial_data_parameter * np.ones((128,128))
    iprint = -1
    if show == True:
        iprint = 100
    bnds = scipy.optimize.Bounds(np.ones(initial_data_parameter.ravel().shape),[np.inf]*len(initial_data_parameter.ravel()))
    optimal = scipy.optimize.minimize(fun=lambda x: data_cost_fn_patch(dsfile,x),jac=lambda x:data_gradient_fn_patch(dsfile,x),x0=initial_data_parameter.ravel(),method='L-BFGS-B',bounds=bnds,options={'iprint':iprint})
    if show == True:
        print(optimal)
    x = optimal.x.reshape(initial_data_parameter.shape)
    optimal_ds = patch_denoise_ds(dsfile,data_parameter=x,reg_parameter=np.ones(x.shape))
    return optimal,optimal_ds