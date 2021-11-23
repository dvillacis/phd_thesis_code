import numpy as np
import pylops, pyproximal
from Operators.Tgamma import Tgamma

from Operators.operators import ActiveOp, InactiveOp
from Operators.TOp import TOp
from Operators.patch import patch, reverse_patch

# DATA GRADIENT

def scalar_data_adjoint(original,reconstruction,data_parameter,show=False):
    nx,ny = original.shape
    n = nx*ny
    K = pylops.Gradient(dims=(nx,ny),kind='forward')
    L = pylops.Diagonal(data_parameter * np.ones(n))
    Id = pylops.Identity(2*n)
    Z = pylops.Zero(n)
    #Act = ActiveOp(reconstruction)
    T = TOp(reconstruction.ravel())
    Inact = InactiveOp(reconstruction)
    A = pylops.Block([[L,K.adjoint()],[T,Inact]])
    b = np.concatenate((reconstruction.ravel()-original.ravel(),np.zeros(2*n)),axis=0)
    p = pylops.optimization.solver.cg(A,b,np.zeros_like(b))
    if show==True:
        print(f'res:{np.linalg.norm(A*p[0]-b)}')
        print(f'cg_out: {p[1:]}')
    return p[0][:n]

def scalar_data_gradient(original,noisy,reconstruction,data_parameter,show=False):
    p = scalar_data_adjoint(original,reconstruction,data_parameter,show=show)
    #L = pylops.Diagonal(p[0])
    #grad = L*(reconstruction.ravel()-noisy.ravel())
    grad = -np.dot(p,reconstruction.ravel()-noisy.ravel())
    return grad

def scalar_data_gradient_ds(ds_denoised,data_parameter):
    grad = 0
    for img in ds_denoised.keys():
        grad += scalar_data_gradient(ds_denoised[img][0],ds_denoised[img][1],ds_denoised[img][2],data_parameter)
    return grad/len(ds_denoised)

# SMOOTH DATA GRADIENT

def smooth_scalar_data_adjoint(original,reconstruction,data_parameter,show=False):
    nx,ny = original.shape
    n = nx*ny
    K = pylops.Gradient(dims=(nx,ny),kind='forward')
    L = pylops.Diagonal(data_parameter * np.ones(n))
    Id = pylops.Identity(2*n)
    Z = pylops.Zero(n)
    Tg = Tgamma(reconstruction.ravel())
    A = pylops.Block([[L,K.adjoint()],[-Tg,Id]])
    b = np.concatenate((reconstruction.ravel()-original.ravel(),np.zeros(2*n)),axis=0)
    #print(f'cond:{A.cond()}')
    p = pylops.optimization.solver.cg(A,b,np.zeros_like(b))
    if show==True:
        print(f'res:{np.linalg.norm(A*p[0]-b)}')
        print(f'cg_out: {p[1:]}')
    return p[0][:n]

def smooth_scalar_data_gradient(original,noisy,reconstruction,data_parameter,show=False):
    p = smooth_scalar_data_adjoint(original,reconstruction,data_parameter,show=show)
    #L = pylops.Diagonal(p[0])
    #grad = L*(reconstruction.ravel()-noisy.ravel())
    grad = -np.dot(p,reconstruction.ravel()-noisy.ravel())
    return grad

def smooth_scalar_data_gradient_ds(ds_denoised,data_parameter):
    grad = 0
    for img in ds_denoised.keys():
        grad += smooth_scalar_data_gradient(ds_denoised[img][0],ds_denoised[img][1],ds_denoised[img][2],data_parameter)
    return grad/len(ds_denoised)


# PATCH

def patch_data_adjoint(original,reconstruction,data_parameter:np.ndarray,show=False):
    data_parameter = patch(data_parameter,original)
    nx,ny = original.shape
    n = nx*ny
    K = pylops.Gradient(dims=(nx,ny),kind='forward')
    L = pylops.Diagonal(data_parameter)
    Id = pylops.Identity(2*n)
    Z = pylops.Zero(n)
    Act = ActiveOp(reconstruction)
    Inact = InactiveOp(reconstruction)
    A = pylops.Block([[L,K.adjoint()],[Act*K-Inact*K,Inact]])
    b = np.concatenate((reconstruction.ravel()-original.ravel(),np.zeros(2*n)),axis=0)
    p = pylops.optimization.solver.cg(A,b,np.zeros_like(b),niter=len(data_parameter)+10)
    if show==True:
        print(p[1:])
    return p[0][:n]

def patch_data_gradient(original,noisy,reconstruction,data_parameter:np.ndarray):
    p = patch_data_adjoint(original,reconstruction,data_parameter)
    L = pylops.Diagonal(p)
    grad = L*(reconstruction.ravel()-noisy.ravel())
    grad = reverse_patch(grad,data_parameter)
    return -grad

def patch_data_gradient_ds(ds_denoised,data_parameter):
    grad = 0
    for img in ds_denoised.keys():
        grad += patch_data_gradient(ds_denoised[img][0],ds_denoised[img][1],ds_denoised[img][2],data_parameter)
    return grad/len(ds_denoised)


# SMOOTH PATCH DATA GRADIENT

def smooth_patch_data_adjoint(original,reconstruction,data_parameter,show=False):
    data_parameter = patch(data_parameter,original)
    nx,ny = original.shape
    n = nx*ny
    K = pylops.Gradient(dims=(nx,ny),kind='forward')
    L = pylops.Diagonal(data_parameter * np.ones(n))
    Id = pylops.Identity(2*n)
    Z = pylops.Zero(n)
    Tg = Tgamma(reconstruction.ravel())
    A = pylops.Block([[L,K.adjoint()],[-Tg,Id]])
    b = np.concatenate((reconstruction.ravel()-original.ravel(),np.zeros(2*n)),axis=0)
    #print(f'cond:{A.cond()}')
    p = pylops.optimization.solver.cg(A,b,np.zeros_like(b))
    if show==True:
        print(f'res:{np.linalg.norm(A*p[0]-b)}')
        print(f'cg_out: {p[1:]}')
    return p[0][:n]

def smooth_patch_data_gradient(original,noisy,reconstruction,data_parameter,show=False):
    p = smooth_patch_data_adjoint(original,reconstruction,data_parameter,show=show)
    L = pylops.Diagonal(p)
    grad = L*(reconstruction.ravel()-noisy.ravel())
    grad = reverse_patch(grad,data_parameter)
    return -grad

def smooth_patch_data_gradient_ds(ds_denoised,data_parameter):
    grad = 0
    for img in ds_denoised.keys():
        grad += smooth_patch_data_gradient(ds_denoised[img][0],ds_denoised[img][1],ds_denoised[img][2],data_parameter)
    return grad/len(ds_denoised)