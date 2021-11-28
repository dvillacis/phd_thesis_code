import numpy as np
import matplotlib.pyplot as plt

from bilearning.Dataset.load_dataset import get_image_pair
from bilearning.TVDenoising.scalar_denoising import denoise
from bilearning.Learning.cost import l2_cost
from bilearning.Learning.reg_gradient import smooth_scalar_reg_gradient, scalar_reg_gradient

orig,noisy = get_image_pair('datasets/cameraman_128_5/filelist.txt',0)
par = np.arange(1e-7,0.08,step=0.09e-2)
grads=[]
smooth_grads = []
costs =[]
for p in par:
    rec = denoise(noisy,1.0,p,niter=5000)
    c = l2_cost(orig,rec)
    gs = smooth_scalar_reg_gradient(orig,noisy,rec,p,show=True)
    g = scalar_reg_gradient(orig,noisy,rec,p,show=True)
    grads.append(g)
    smooth_grads.append(gs)
    print(f'p: {p:.4f}, c:{c:.7f}, g:{g}')
    costs.append(c)
fig,ax = plt.subplots()
ax.plot(par,grads,color='red')
ax.plot(par,smooth_grads,color='green')
ax2 = ax.twinx()
ax2.plot(par,costs)
ax.grid()
plt.show()