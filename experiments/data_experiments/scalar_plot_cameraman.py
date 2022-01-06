
import numpy as np
import pandas as pd

import os
import sys
sys.path.append('../../')

from bilearning.Dataset.load_dataset import get_image_pair
from bilearning.TVDenoising.scalar_denoising import denoise
from bilearning.Learning.cost import l2_cost

out_dir = 'scalar_plot_cameraman'
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

orig, noisy = get_image_pair('../../datasets/cameraman_128_5/filelist.txt', 0)
par = np.arange(1.0, 501.0, step=1.0)
costs = []
for p in par:
    rec = denoise(noisy, p, 1.0, niter=1000)
    c = l2_cost(orig, rec)
    print(f'p: {p}, c:{c}')
    costs.append(c)
out_df = pd.DataFrame(columns=['par','cost'])
out_df['par'] = par
out_df['cost'] = costs
print(out_df)
out_df.to_csv(os.path.join(out_dir,'scalar_plot.csv'))

alpha0 = 15
print(f'Executing the scalar cameraman experiment with lambda={alpha0}')
cmd = f'python ../../data_learning.py ../../datasets/cameraman_128_5/filelist.txt -t scalar -i {str(alpha0)} -o {os.path.join(out_dir,str(alpha0))} -v'
os.system(cmd)
