import numpy as np
from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator


class SDGP(ProxOperator):
    def __init__(self, b, sigma1: np.array, sigma2:np.array):
        super().__init__(None, True)
        self.b = b
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def __call__(self, x):
        x = np.where(x<=0,1e-10,x)
        return 0.5 * np.dot(self.sigma1, (x-self.b)**2) + np.dot(self.sigma2, (x-self.b*np.log(x)))

    @_check_tau
    def prox(self, x, tau):
        num1 = x + tau * self.sigma1 * self.b - tau * self.sigma2
        num2 = np.sqrt((tau * self.sigma2 - x - tau * self.sigma1 * self.b)**2 + 4 * (1 + tau * self.sigma1) * (tau * self.sigma2 * self.b))
        x = (num1+num2) / (2. + 2 * tau * self.sigma1)
        #print(f'prox:{x}')
        return x

    def grad(self, x):
        g = self.sigma1 * (x - self.b) + self.sigma2 * (1 - self.b/x)
        return g
