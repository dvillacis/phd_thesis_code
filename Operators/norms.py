
import numpy as np
from pylops import FirstDerivative

def pointwise_euclidean_norm(u):
    n = u.shape
    if len(n)>1:
        raise ValueError('Input must be a 1D vector...')
    u = np.reshape(u,(n[0]//2,2),order='F')
    nu = np.linalg.norm(u,axis=1)
    return nu

def tv_smooth_subdiff(u,gamma=1000):
    n = len(u)
    Kx = FirstDerivative(n,kind='forward',dir=0,edge=True)
    Ky = FirstDerivative(n,kind='forward',dir=1,edge=True)
    Kxu = Kx*u
    Kyu = Ky*u
    nu = np.linalg.norm(np.vstack((Kxu,Kyu)).T,axis=1)
    a = np.where(gamma*nu-1 <= -1/(2*gamma),gamma,1./nu)
    return a*Kxu,a*Kyu