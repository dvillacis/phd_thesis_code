
import numpy as np

def pointwise_euclidean_norm(u):
    n = u.shape
    if len(n)>1:
        raise ValueError('Input must be a 1D vector...')
    u = np.reshape(u,(n[0]//2,2),order='F')
    nu = np.linalg.norm(u,axis=1)
    return nu