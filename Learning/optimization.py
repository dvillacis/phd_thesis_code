import scipy
from Learning.cost import l2_cost_ds
from TVDenoising.scalar_denoising import denoise_ds
from Learning.data_gradient import scalar_data_gradient_ds

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