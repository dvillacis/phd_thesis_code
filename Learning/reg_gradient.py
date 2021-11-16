import numpy as np
import pylops
from pylops import FirstDerivative

from Operators.operators import ActiveOp, InactiveOp
from Operators.TOp import TOp
from Operators.patch import patch, reverse_patch
from Operators.norms import pointwise_euclidean_norm
from Operators.Tgamma import Tgamma

def scalar_reg_adjoint(original,reconstruction,reg_parameter,show=False):
    nx,ny = original.shape
    n = nx*ny
    K = pylops.Gradient(dims=(nx,ny),kind='forward')
    Id = pylops.Identity(n)
    L = pylops.Diagonal(reg_parameter * np.ones(2*n))
    Id2 = pylops.Identity(2*n)
    Z = pylops.Zero(n)
    Act = ActiveOp(reconstruction)
    T = TOp(reconstruction.ravel())
    Inact = InactiveOp(reconstruction)
    A = pylops.Block([[Id,K.adjoint()],[-L*Inact*T,Inact+1e-5*Act]])
    b = np.concatenate((reconstruction.ravel()-original.ravel(),np.zeros(2*n)),axis=0)
    p = pylops.optimization.solver.cg(A,b,np.zeros_like(b),tol=1e-4)
    if show==True:
        print(f'res:{np.linalg.norm(A*p[0]-b)}')
        print(f'cg_out: {p[1:]}')
    adj = p[0][:n]
    return adj

def scalar_reg_gradient(original,noisy,reconstruction,reg_parameter,show=False,tol=1e-7):
    nx,ny = original.shape
    n = nx*ny
    p = scalar_reg_adjoint(original,reconstruction,reg_parameter,show=show)
    Kx = FirstDerivative(n,kind='centered',dir=0,edge=True)
    Ky = FirstDerivative(n,kind='centered',dir=1,edge=True)
    Kxu = Kx*reconstruction.ravel()
    Kyu = Ky*reconstruction.ravel()
    Kxp = Kx*p
    Kyp = Ky*p
    nKu = np.linalg.norm(np.vstack((Kxu,Kyu)).T,axis=1)
    mul = np.where(nKu<tol,0,1/nKu)
    grad = mul * (Kxu * Kxp + Kyu * Kyp)
    return -np.sum(grad)

def scalar_reg_gradient_ds(ds_denoised,reg_parameter):
    grad = 0
    for img in ds_denoised.keys():
        grad += scalar_reg_gradient(ds_denoised[img][0],ds_denoised[img][1],ds_denoised[img][2],reg_parameter)
    return grad/len(ds_denoised)

# SMOOTH SCALAR
def smooth_scalar_reg_adjoint(original,reconstruction,reg_parameter,show=False):
    nx,ny = original.shape
    n = nx*ny
    K = pylops.Gradient(dims=(nx,ny),kind='forward')
    L = pylops.Diagonal((reg_parameter) * np.ones(2*n))
    Id = pylops.Identity(2*n)
    Idn = pylops.Identity(n)
    Z = pylops.Zero(n)
    Tg = Tgamma(reconstruction.ravel(),gamma=1e-7)
    A = pylops.Block([[Idn,K.adjoint()],[-L*Tg,Id]])
    b = np.concatenate(((reconstruction.ravel()-original.ravel()),np.zeros(2*n)),axis=0)
    p = pylops.optimization.solver.cg(A,b,np.zeros_like(b),niter=10)
    if show==True:
        print(f'cg_out: {p[1:]}')
        print(f'res:{np.linalg.norm(A*p[0]-b)}')
    return p[0][:n]

def smooth_scalar_reg_gradient(original,noisy,reconstruction,reg_parameter,gamma=1e-4,show=False):
    nx,ny = original.shape
    n = nx*ny
    p = smooth_scalar_reg_adjoint(original,reconstruction,reg_parameter,show=show)
    Kx = FirstDerivative(n,kind='centered',dir=0,edge=True)
    Ky = FirstDerivative(n,kind='centered',dir=1,edge=True)
    Kxu = Kx*reconstruction.ravel()
    Kyu = Ky*reconstruction.ravel()
    Kxp = Kx*p
    Kyp = Ky*p
    nKu = np.linalg.norm(np.vstack((Kxu,Kyu)).T,axis=1)
    mul = np.where(nKu<gamma,nKu/(2*gamma**2)-2/gamma,1/nKu)
    hx = mul*Kxu
    hy = mul*Kyu
    #grad = L*(reconstruction.ravel()-noisy.ravel())
    grad = -np.sum(hx*Kxp + hy*Kyp)
    return grad

# PATCH
def patch_reg_adjoint(original,reconstruction,reg_parameter:np.ndarray,show=False):
    reg_parameter = patch(reg_parameter,original)
    nx,ny = original.shape
    n = nx*ny
    K = pylops.Gradient(dims=(nx,ny),kind='forward')
    Id = pylops.Identity(n)
    L = pylops.Diagonal(np.concatenate((reg_parameter,reg_parameter)))
    Id2 = pylops.Identity(2*n)
    Z = pylops.Zero(n)
    Act = ActiveOp(reconstruction)
    T = TOp(reconstruction.ravel())
    Inact = InactiveOp(reconstruction)
    A = pylops.Block([[Id,K.adjoint()],[-L*T,Inact + 1*Act]])
    b = np.concatenate((reconstruction.ravel()-original.ravel(),np.zeros(2*n)),axis=0)
    p = pylops.optimization.solver.cg(A,b,np.zeros_like(b),tol=1e-4)
    if show==True:
        print(f'res:{np.linalg.norm(A*p[0]-b)}')
        print(f'cg_out: {p[1:]}')
    adj = p[0][:n]
    return adj

def patch_reg_gradient(original,noisy,reconstruction,reg_parameter:np.ndarray,tol=1e-7):
    p = patch_reg_adjoint(original,reconstruction,reg_parameter)
    nx,ny = original.shape
    n = nx*ny
    Kx = FirstDerivative(n,kind='centered',dir=0,edge=True)
    Ky = FirstDerivative(n,kind='centered',dir=1,edge=True)
    Kxu = Kx*reconstruction.ravel()
    Kyu = Ky*reconstruction.ravel()
    Kxp = Kx*p
    Kyp = Ky*p
    nKu = np.linalg.norm(np.vstack((Kxu,Kyu)).T,axis=1)
    mul = np.where(nKu<tol,0,1/nKu)
    grad = mul * (Kxu * Kxp + Kyu * Kyp)
    grad = reverse_patch(grad,reg_parameter)
    return grad

def patch_reg_gradient_ds(ds_denoised,reg_parameter):
    grad = 0
    for img in ds_denoised.keys():
        grad += patch_reg_gradient(ds_denoised[img][0],ds_denoised[img][1],ds_denoised[img][2],reg_parameter)
    return grad/len(ds_denoised)