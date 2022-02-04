from PIL.Image import init
import numpy as np
import scipy
from bilearning.Learning.cost import l2_cost_ds
from bilearning.TVDenoising.scalar_denoising import denoise_ds
from bilearning.TVDenoising.patch_denoising import patch_denoise_ds
from bilearning.Learning.data_gradient import scalar_data_gradient_ds, patch_data_gradient_ds, smooth_scalar_data_gradient_ds, smooth_patch_data_gradient_ds
from bilearning.Learning.reg_gradient import scalar_reg_gradient_ds, patch_reg_gradient_ds, smooth_scalar_reg_gradient_ds, smooth_patch_reg_gradient_ds
from bilearning.TrustRegion.nsdogbox import nsdogbox
from bilearning.Operators.patch import OnesPatch, Patch

#################################
# SCALAR DATA PARAMETER LEARNING
#################################

def data_cost_fn_scalar(dsfile,data_parameter):
    den_ds = denoise_ds(dsfile,data_parameter=data_parameter.data[0],reg_parameter=1.0,niter=1000)
    return l2_cost_ds(den_ds)

def data_gradient_fn_scalar(dsfile,data_parameter):
    den_ds = denoise_ds(dsfile,data_parameter=data_parameter.data[0],reg_parameter=1.0,niter=1000)
    return np.array([scalar_data_gradient_ds(den_ds,data_parameter.data[0])])

def smooth_data_gradient_fn_scalar(dsfile,data_parameter):
    den_ds = denoise_ds(dsfile,data_parameter=data_parameter.data[0],reg_parameter=1.0,niter=1000)
    return np.array([smooth_scalar_data_gradient_ds(den_ds,data_parameter.data[0])])

def find_optimal_data_scalar(dsfile,initial_data_parameter,show=False):
    iprint = 0
    if show == True:
        iprint = 2
    #bnds = scipy.optimize.Bounds(0.001,np.inf)
    optimal = nsdogbox(fun=lambda x: data_cost_fn_scalar(dsfile,x),grad=lambda x:data_gradient_fn_scalar(dsfile,x),reg_grad=lambda x:smooth_data_gradient_fn_scalar(dsfile,x),x0=initial_data_parameter,verbose=iprint,initial_radius=1.0)
    if show == True:
        print(optimal)
    optimal_ds = denoise_ds(dsfile,data_parameter=optimal.x.data[0],reg_parameter=1.0)
    return optimal,optimal_ds

#################################
# SCALAR REGULARIZATION PARAMETER LEARNING
#################################

def reg_cost_fn_scalar(dsfile,reg_parameter):
    den_ds = denoise_ds(dsfile,data_parameter=1.0,reg_parameter=reg_parameter,niter=100)
    return l2_cost_ds(den_ds)

def reg_gradient_fn_scalar(dsfile,reg_parameter):
    den_ds = denoise_ds(dsfile,data_parameter=1.0,reg_parameter=reg_parameter[0],niter=100)
    return np.array([scalar_reg_gradient_ds(den_ds,reg_parameter[0])])

def smooth_reg_gradient_fn_scalar(dsfile,reg_parameter,gamma=100000):
    den_ds = denoise_ds(dsfile,data_parameter=1.0,reg_parameter=reg_parameter[0],niter=100)
    return np.array([smooth_scalar_reg_gradient_ds(den_ds,reg_parameter[0],gamma=gamma)])

def find_optimal_reg_scalar(dsfile,initial_reg_parameter,show=False,gamma=100000,threshold_radius=1e-4):
    iprint = 0
    if show == True:
        iprint = 2
    optimal = nsdogbox(fun=lambda x: reg_cost_fn_scalar(dsfile,x),grad=lambda x:reg_gradient_fn_scalar(dsfile,x),reg_grad=lambda x:smooth_reg_gradient_fn_scalar(dsfile,x,gamma=gamma),x0=np.array([initial_reg_parameter]),verbose=iprint,initial_radius=0.001,threshold_radius=threshold_radius)
    if show == True:
        print(optimal)
    optimal_ds = denoise_ds(dsfile,data_parameter=1.0,reg_parameter=optimal.x)
    return optimal,optimal_ds

#################################
# PATCH DATA PARAMETER LEARNING
#################################

def data_cost_fn_patch(dsfile,data_parameter:Patch):
    den_ds = patch_denoise_ds(dsfile,data_parameter=data_parameter,reg_parameter=OnesPatch(data_parameter.px,data_parameter.py),niter=2000, show=False)
    return l2_cost_ds(den_ds)

def data_gradient_fn_patch(dsfile,data_parameter:Patch):
    den_ds = patch_denoise_ds(dsfile,data_parameter=data_parameter,reg_parameter=OnesPatch(data_parameter.px,data_parameter.py),niter=2000, show=False)
    grad = patch_data_gradient_ds(den_ds,data_parameter)
    return grad

def smooth_data_gradient_fn_patch(dsfile,data_parameter:Patch):
    den_ds = patch_denoise_ds(dsfile,data_parameter=data_parameter,reg_parameter=OnesPatch(data_parameter.px,data_parameter.py),niter=2000, show=False)
    grad = smooth_patch_data_gradient_ds(den_ds,data_parameter)
    return grad

def find_optimal_data_patch(dsfile,initial_data_parameter:Patch,show=False):
    iprint = 0
    if show == True:
        iprint = 2

    optimal = nsdogbox(fun=lambda x: data_cost_fn_patch(dsfile,x),grad=lambda x:data_gradient_fn_patch(dsfile,x),reg_grad=lambda x:smooth_data_gradient_fn_patch(dsfile,x),x0=initial_data_parameter,verbose=iprint,initial_radius=np.linalg.norm(initial_data_parameter.data,np.inf),max_radius=1000)
    # np.linalg.norm(initial_data_parameter.data,np.inf)

    if show == True:
        print(optimal)
    x = optimal.x
    optimal_ds = patch_denoise_ds(dsfile, data_parameter=x, reg_parameter=OnesPatch(x.px, x.py),niter=2000, show=True)
    return optimal,optimal_ds

#################################
# PATCH REGULARIZATION PARAMETER LEARNING
#################################

def reg_cost_fn_patch(dsfile,reg_parameter):
    den_ds = patch_denoise_ds(dsfile,data_parameter=np.ones(reg_parameter.shape),reg_parameter=reg_parameter,niter=3000)
    return l2_cost_ds(den_ds)

def reg_gradient_fn_patch(dsfile,reg_parameter):
    den_ds = patch_denoise_ds(dsfile,data_parameter=np.ones(reg_parameter.shape),reg_parameter=reg_parameter,niter=3000)
    return patch_reg_gradient_ds(den_ds,reg_parameter)

def smooth_reg_gradient_fn_patch(dsfile,reg_parameter):
    den_ds = patch_denoise_ds(dsfile,data_parameter=np.ones(reg_parameter.shape),reg_parameter=reg_parameter,niter=3000)
    return smooth_patch_reg_gradient_ds(den_ds,reg_parameter)

def find_optimal_reg_patch(dsfile,initial_reg_parameter,show=False):
    iprint = 0
    if show == True:
        iprint = 2
    #bnds = scipy.optimize.Bounds(0.001*np.ones(initial_reg_parameter.ravel().shape),[np.inf]*len(initial_reg_parameter.ravel()))
    #optimal = scipy.optimize.minimize(fun=lambda x: reg_cost_fn_patch(dsfile,x),jac=lambda x:reg_gradient_fn_patch(dsfile,x),hess=scipy.optimize.SR1(),x0=initial_reg_parameter.ravel(),method='trust-constr',bounds=bnds,options={'verbose':iprint,'gtol':1e-6})
    optimal = nsdogbox(fun=lambda x: reg_cost_fn_patch(dsfile,x),grad=lambda x:reg_gradient_fn_patch(dsfile,x),reg_grad=lambda x:smooth_reg_gradient_fn_patch(dsfile,x),x0=initial_reg_parameter.ravel(),verbose=iprint,initial_radius=0.01)
    if show == True:
        print(optimal)
    x = optimal.x
    optimal_ds = patch_denoise_ds(dsfile,data_parameter=np.ones(x.shape),reg_parameter=x,niter=1000)
    return optimal,optimal_ds