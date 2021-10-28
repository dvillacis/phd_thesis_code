import numpy as np
from pylops import LinearOperator
from Operators.norms import pointwise_euclidean_norm

def phi(u,gamma):
    nu = pointwise_euclidean_norm(u)
    return np.concatenate((np.where(nu<gamma,-nu/gamma**2+2/gamma,1/nu),np.where(nu<gamma,-nu/gamma**2+2/gamma,1/nu)))

def psi(u,gamma):
    nu = pointwise_euclidean_norm(u)
    return np.concatenate((np.where(nu<gamma,-1/nu/gamma**2,-1/nu**3),np.where(nu<gamma,-1/nu/gamma**2,-1/nu**3)))

class Tgamma(LinearOperator):
    def __init__(self, u, gamma=1e-3, dtype='float64'):
        self.u = u
        self.phi_u = phi(u,gamma)
        self.psi_u = psi(u,gamma)
        self.shape = (len(u),len(u))
        self.dtype = dtype

    def _matvec(self,x):
        return self.phi_u * x + self.psi_u * self.u * self.u * x