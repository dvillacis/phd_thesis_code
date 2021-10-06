import numpy as np

def patch(x,out):
    nx,ny = out.shape
    px = int(np.sqrt(len(x)))
    x = x.reshape((px,px))
    x = np.kron(x,np.ones((nx//px,ny//px)))
    return x

def reverse_patch(x,out):
    nx = int(np.sqrt(len(x)))
    x = x.reshape((nx,nx))
    px = nx // int(np.sqrt(len(out)))
    result = np.add.reduceat(np.add.reduceat(x, np.arange(0, x.shape[0], px), axis=0),np.arange(0, x.shape[1], px), axis=1)
    return result / px