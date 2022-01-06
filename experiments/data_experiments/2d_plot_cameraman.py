import pandas as pd
import numpy as np
import os
import sys

sys.path.append('../../')

from bilearning.Operators.patch import OnesPatch, Patch
from bilearning.Dataset.load_dataset import get_image_pair
from bilearning.TVDenoising.patch_denoising import patch_denoise
from bilearning.Learning.cost import l2_cost

out_dir = '2d_plot_cameraman'
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

orig, noisy = get_image_pair('../../datasets/cameraman_128_5/filelist.txt', 0)
par = np.arange(1.0, 101.0, step=1.0)
costs = []
p1s = []
p2s = []
for p1 in par:
    for p2 in par:
        p = Patch(np.array([p1,p2]),2,1)
        o = OnesPatch(2,1)
        rec = patch_denoise(noisy, p, o, niter=1000)
        c = l2_cost(orig, rec)
        print(f'p: {p}, c:{c}')
        costs.append(c)
        p1s.append(p1)
        p2s.append(p2)
out_df = pd.DataFrame(columns=['par1','par2','cost'])
out_df['par1'] = p1s
out_df['par2'] = p2s
out_df['cost'] = costs
print(out_df)
out_df.to_csv(os.path.join(out_dir,'scalar_plot.csv'))

alpha0 = 15
print(f'Executing the 2d patch cameraman experiment with lambda={alpha0}')
cmd = f'python ../../data_learning.py ../../datasets/cameraman_128_5/filelist.txt -t patch -prows 2 -pdata 15 15 -o {os.path.join(out_dir,str(alpha0))} -v'
os.system(cmd)