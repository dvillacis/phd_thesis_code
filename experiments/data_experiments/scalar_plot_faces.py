import numpy as np
import pandas as pd

import os
import sys
sys.path.append('../../')

from bilearning.Dataset.load_dataset import get_image_pair, load_ds_file
from bilearning.TVDenoising.scalar_denoising import denoise_ds
from bilearning.Learning.cost import l2_cost_ds


out_dir = 'scalar_plot_faces'
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

ds = '../../datasets/faces_train_128_10/filelist.txt'
par = np.arange(1.0, 501.0, step=1.0)
costs = []
for p in par:
    rec = denoise_ds(ds, p, 1.0, niter=1000)
    c = l2_cost_ds(rec)
    print(f'p: {p}, c:{c}')
    costs.append(c)
out_df = pd.DataFrame(columns=['par', 'cost'])
out_df['par'] = par
out_df['cost'] = costs
print(out_df)
out_df.to_csv(os.path.join(out_dir, 'scalar_plot.csv'))

alpha0 = 15
print(f'Executing the scalar faces experiment with lambda={alpha0}')
cmd = f'python ../../data_learning.py ../../datasets/faces_train_128_10/filelist.txt -t scalar -i {str(alpha0)} -o {os.path.join(out_dir,str(alpha0))} -v'
os.system(cmd)
