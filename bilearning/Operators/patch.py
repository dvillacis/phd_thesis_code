import numpy as np
from pyproximal import proximal

class Patch:
    def __init__(self,data,px,py) -> None:
        self.px = px
        self.py = py
        self.data = data
    def copy(self):
        return Patch(self.data,self.px,self.py)
    def get_matrix(self):
        return self.data.reshape((self.px,self.py))
    def map_to_img(self,img):
        nx,ny = img.shape
        m = self.get_matrix()
        m = np.kron(m,np.ones((nx//self.px,ny//self.py)))
        return m.ravel()
    def reduce_from_img(self,img):
        nx,ny = img.shape
        mx = nx//self.px
        my = ny//self.py
        result = np.add.reduceat(np.add.reduceat(img, np.arange(0, img.shape[0], mx), axis=0),np.arange(0, img.shape[1], my), axis=1)
        #return (result/(mx*my)).ravel()
        return result.ravel()
    def __str__(self) -> str:
        return f'Patch ({self.data},{self.px},{self.py})'
    def __repr__(self) -> str:
        return f'Patch ({self.data},{self.px},{self.py})'

class OnesPatch(Patch):
    def __init__(self, px, py) -> None:
        data = np.ones((px,py))
        super().__init__(data, px, py)


def patch(x,out):
    nx,ny = out.shape
    px = int(np.sqrt(len(x)))
    x = x.reshape((px,px))
    x = np.kron(x,np.ones((nx//px,ny//px)))
    return x.ravel()

def reverse_patch(x,out):
    nx = int(np.sqrt(len(x)))
    x = x.reshape((nx,nx))
    px = nx // int(np.sqrt(len(out)))
    result = np.add.reduceat(np.add.reduceat(x, np.arange(0, x.shape[0], px), axis=0),np.arange(0, x.shape[1], px), axis=1)
    return (result / px).ravel()