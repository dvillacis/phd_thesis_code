import numpy as np
from pylops import LinearOperator, FirstDerivative

def phi(nu,tol):
    return np.where(nu<tol,1,1/nu)

def psi(nu,tol):
    return np.where(nu<tol,0,-1/nu**3)

class TOp(LinearOperator):
    def __init__(self, u, tol=1e-10, dtype='float64'):
        n = len(u)
        self.Kx = FirstDerivative(n,kind='centered',dir=0,edge=True)
        self.Ky = FirstDerivative(n,kind='centered',dir=1,edge=True)
        self.Kxu = self.Kx*u
        self.Kyu = self.Ky*u
        self.Kxu2 = self.Kxu * self.Kxu
        self.Kyu2 = self.Kyu * self.Kyu
        self.Kxyu = self.Kxu * self.Kyu
        self.nKu = np.linalg.norm(np.vstack((self.Kxu,self.Kyu)).T,axis=1)
        self.phi_u = phi(self.nKu,tol)
        self.psi_u = psi(self.nKu,tol)
        self.shape = (2*len(u),len(u))
        self.dtype = dtype

    def _matvec(self,x):
        Kxx = self.Kx * x
        Kyx = self.Ky * x
        a = self.psi_u * (self.Kxu2 * Kxx + self.Kxyu * Kyx) + self.phi_u * Kxx
        b = self.psi_u * (self.Kyu2 * Kyx + self.Kxyu * Kxx) + self.phi_u * Kyx
        return np.concatenate((a,b))