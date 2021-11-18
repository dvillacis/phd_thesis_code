import numpy as np
import scipy
from Learning.cost import l2_cost_ds
from TVDenoising.scalar_denoising import denoise_ds
from TVDenoising.patch_denoising import patch_denoise_ds
from Learning.data_gradient import scalar_data_gradient_ds, patch_data_gradient_ds
from Learning.reg_gradient import scalar_reg_gradient_ds, patch_reg_gradient_ds, smooth_scalar_reg_gradient_ds

#################################
# SCALAR DATA PARAMETER LEARNING
#################################

def data_cost_fn_scalar(dsfile,data_parameter):
    den_ds = denoise_ds(dsfile,data_parameter=data_parameter,reg_parameter=1.0,niter=100)
    return l2_cost_ds(den_ds)

def data_gradient_fn_scalar(dsfile,data_parameter):
    den_ds = denoise_ds(dsfile,data_parameter=data_parameter,reg_parameter=1.0,niter=100)
    return scalar_data_gradient_ds(den_ds,data_parameter)

def find_optimal_data_scalar(dsfile,initial_data_parameter,show=False):
    iprint = 0
    if show == True:
        iprint = 3
    bnds = scipy.optimize.Bounds(0.001,np.inf)
    optimal = scipy.optimize.minimize(fun=lambda x: data_cost_fn_scalar(dsfile,x),hess=scipy.optimize.SR1(),jac=lambda x:data_gradient_fn_scalar(dsfile,x),x0=initial_data_parameter,method='trust-constr',bounds=bnds,options={'verbose':iprint})
    if show == True:
        print(optimal)
    optimal_ds = denoise_ds(dsfile,data_parameter=optimal.x,reg_parameter=1.0)
    return optimal,optimal_ds

#################################
# SCALAR REGULARIZATION PARAMETER LEARNING
#################################

def reg_cost_fn_scalar(dsfile,reg_parameter):
    den_ds = denoise_ds(dsfile,data_parameter=1.0,reg_parameter=reg_parameter,niter=5000)
    return l2_cost_ds(den_ds)

def reg_gradient_fn_scalar(dsfile,reg_parameter):
    den_ds = denoise_ds(dsfile,data_parameter=1.0,reg_parameter=reg_parameter,niter=5000)
    return smooth_scalar_reg_gradient_ds(den_ds,reg_parameter)

def find_optimal_reg_scalar(dsfile,initial_reg_parameter,show=False):
    iprint = -1
    if show == True:
        iprint = 100
    bnds = scipy.optimize.Bounds(1e-10,np.inf)
    optimal = scipy.optimize.minimize(fun=lambda x: reg_cost_fn_scalar(dsfile,x),jac=lambda x:reg_gradient_fn_scalar(dsfile,x),hess=scipy.optimize.SR1(),x0=initial_reg_parameter,method='trust-constr',bounds=bnds,options={'verbose':iprint,'gtol':1e-6})
    if show == True:
        print(optimal)
    optimal_ds = denoise_ds(dsfile,data_parameter=1.0,reg_parameter=optimal.x)
    return optimal,optimal_ds

#################################
# PATCH DATA PARAMETER LEARNING
#################################

def data_cost_fn_patch(dsfile,data_parameter:np.ndarray):
    den_ds = patch_denoise_ds(dsfile,data_parameter=data_parameter,reg_parameter=np.ones(data_parameter.shape),niter=5000)
    return l2_cost_ds(den_ds)

def data_gradient_fn_patch(dsfile,data_parameter:np.ndarray):
    den_ds = patch_denoise_ds(dsfile,data_parameter=data_parameter,reg_parameter=np.ones(data_parameter.shape),niter=5000)
    grad = patch_data_gradient_ds(den_ds,data_parameter)
    return grad

def find_optimal_data_patch(dsfile,initial_data_parameter,show=False):
    iprint = 0
    if show == True:
        iprint = 3
    bnds = scipy.optimize.Bounds(0.001*np.ones(initial_data_parameter.ravel().shape),[np.inf]*len(initial_data_parameter.ravel()))
    optimal = scipy.optimize.minimize(fun=lambda x: data_cost_fn_patch(dsfile,x),jac=lambda x:data_gradient_fn_patch(dsfile,x),x0=initial_data_parameter.ravel(),hess=scipy.optimize.SR1(),method='trust-constr',bounds=bnds,options={'verbose':iprint,'gtol':1e-4})
    if show == True:
        print(optimal)
    x = optimal.x
    optimal_ds = patch_denoise_ds(dsfile,data_parameter=x,reg_parameter=np.ones(x.shape))
    return optimal,optimal_ds

#################################
# PATCH REGULARIZATION PARAMETER LEARNING
#################################

def reg_cost_fn_patch(dsfile,reg_parameter):
    den_ds = patch_denoise_ds(dsfile,data_parameter=np.ones(reg_parameter.shape),reg_parameter=reg_parameter,niter=5000)
    return l2_cost_ds(den_ds)

def reg_gradient_fn_patch(dsfile,reg_parameter):
    den_ds = patch_denoise_ds(dsfile,data_parameter=np.ones(reg_parameter.shape),reg_parameter=reg_parameter,niter=5000)
    return patch_reg_gradient_ds(den_ds,reg_parameter)

def find_optimal_reg_patch(dsfile,initial_reg_parameter,show=False):
    iprint = -1
    if show == True:
        iprint = 100
    bnds = scipy.optimize.Bounds(0.001*np.ones(initial_reg_parameter.ravel().shape),[np.inf]*len(initial_reg_parameter.ravel()))
    optimal = scipy.optimize.minimize(fun=lambda x: reg_cost_fn_patch(dsfile,x),jac=lambda x:reg_gradient_fn_patch(dsfile,x),hess=scipy.optimize.SR1(),x0=initial_reg_parameter.ravel(),method='trust-constr',bounds=bnds,options={'verbose':iprint,'gtol':1e-6})
    if show == True:
        print(optimal)
    x = optimal.x
    optimal_ds = patch_denoise_ds(dsfile,data_parameter=np.ones(x.shape),reg_parameter=x)
    return optimal,optimal_ds