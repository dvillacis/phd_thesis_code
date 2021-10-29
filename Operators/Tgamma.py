import numpy as np
from pylops import LinearOperator, FirstDerivative

def phi(nu,gamma):
    return np.where(nu<gamma,-nu/gamma**2+2/gamma,1/nu)

def psi(nu,gamma):
    return np.where(nu<gamma,-1/nu/gamma**2,-1/nu**3)

class Tgamma(LinearOperator):
    def __init__(self, u, gamma=1e-3, dtype='float64'):
        n = len(u)
        self.Kx = FirstDerivative(n,kind='centered',dir=0,edge=True)
        self.Ky = FirstDerivative(n,kind='centered',dir=1,edge=True)
        self.Kxu = self.Kx*u
        self.Kyu = self.Ky*u
        self.Kxu2 = self.Kxu * self.Kxu
        self.Kyu2 = self.Kyu * self.Kyu
        self.Kxyu = self.Kxu * self.Kyu
        self.nKu = np.linalg.norm(np.vstack((self.Kxu,self.Kyu)).T,axis=1)
        print(self.nKu)
        self.phi_u = phi(self.nKu,gamma)
        self.psi_u = psi(self.nKu,gamma)
        print(f'phi_u:\n{self.phi_u}\npsi_u:\n{self.psi_u}')
        self.shape = (2*len(u),len(u))
        self.dtype = dtype

    def _matvec(self,x):
        Kxx = self.Kx * x
        Kyx = self.Ky * x
        print(self.phi_u * Kxx)
        print(self.psi_u * (self.Kxu2 * Kxx + self.Kxyu * Kyx))
        a = self.psi_u * (self.Kxu2 * Kxx + self.Kxyu * Kyx) + self.phi_u * Kxx
        b = self.psi_u * (self.Kyu2 * Kyx + self.Kxyu * Kxx) + self.phi_u * Kyx
        return np.concatenate((a,b))