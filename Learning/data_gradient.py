import numpy as np
import pylops, pyproximal

from Operators.operators import ActiveOp, InactiveOp

def scalar_data_adjoint(original,reconstruction,data_parameter,show=False):
    nx,ny = original.shape
    n = nx*ny
    K = pylops.Gradient(dims=(nx,ny),kind='forward')
    L = pylops.Diagonal(data_parameter * np.ones(n))
    Id = pylops.Identity(2*n)
    Z = pylops.Zero(n)
    Act = ActiveOp(reconstruction)
    Inact = InactiveOp(reconstruction)
    A = pylops.Block([[L,K.adjoint()],[Act*K-Inact*K,Inact]])
    b = np.concatenate((-reconstruction.ravel()+original.ravel(),np.zeros(2*n)),axis=0)
    p = pylops.optimization.solver.cg(A,b,np.zeros_like(b))
    if show==True:
        print(p[1:])
    return p[0][:n]

def scalar_data_gradient(original,noisy,reconstruction,data_parameter):
    p = scalar_data_adjoint(original,reconstruction,data_parameter)
    #L = pylops.Diagonal(p[0])
    #grad = L*(reconstruction.ravel()-noisy.ravel())
    grad = np.dot(p,reconstruction.ravel()-noisy.ravel())
    return grad

def scalar_data_gradient_ds(ds_denoised,data_parameter):
    grad = 0
    for img in ds_denoised.keys():
        grad += scalar_data_gradient(ds_denoised[img][0],ds_denoised[img][1],ds_denoised[img][2],data_parameter)
    return grad/len(ds_denoised)