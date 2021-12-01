import numpy as np
import os
import sys
from ast import literal_eval

alpha0 = 0.01

out_dir = 'scalar_cameraman_reg'

#out_dir = os.path.join(out_dir,str(patch_size))
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

print(f'Executing the scalar cameraman experiment')
cmd = f'python ../regularization_learning.py ../datasets/cameraman_128_5/filelist.txt -t scalar -i {str(alpha0)} -o {os.path.join(out_dir,str(alpha0))} -v'
os.system(cmd)