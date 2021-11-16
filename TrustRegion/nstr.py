import numpy as np
from numpy.core.numeric import allclose, zeros_like
import scipy
import scipy.linalg
from scipy.linalg import solve, norm
from scipy.optimize import BFGS


def model(f,g,B,x,p,radius):
    if norm(p) > radius:
        raise ArithmeticError(f'p must not be bigger than the radius. p:{p}, radius:{radius}')
    return f(x) + np.dot(g(x),p) + 0.5*np.dot(p,B.dot(p))

def nstr(f,g,x0,step_finder,init_radius = 1.0,maxiter = 100, tol=1e-5, max_radius=10.0, verbose=False):
    x = x0
    B = BFGS()
    if isinstance(x,float):
        B.initialize(1,'hess')
    elif isinstance(x,np.ndarray):
        B.initialize(len(x),'hess')
    radius = init_radius
    it = 0
    if verbose: print('it\tfx\tgx\tBx\tradius')
    while it < maxiter:
        it += 1
        p = step_finder(g(x),B,radius)
        rho = (f(x) - f(x+p))/(model(f,g,B,x,p,radius)-model(f,g,B,x+p,p,radius))
        if rho < 0.25:
            radius = 0.25 * radius
        elif rho >= 0.75:
            radius = min(2*radius,max_radius)

        if rho > 0.9:
            B.update(p,g(x)-g(x+p))
            x = x+p
        elif np.allclose(p,np.zeros(p.shape),1e-10):
            x = x+p
            break
            
        
        if verbose: 
            print(x,p,f(x))
            print(f'{it}\t{f(x):.6f}\t{g(x):.6f}\t{B.get_matrix()}\t{radius}')
        if np.linalg.norm(g(x))<tol: 
            return x,f(x),it
            break
        
    return x,f(x),it
