import numpy as np
import os
import sys
from ast import literal_eval

alpha0 = 0.03

out_dir = 'scalar_faces_reg_10'

#out_dir = os.path.join(out_dir,str(patch_size))
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

print(f'Executing the scalar faces experiment')
cmd = f'python ../regularization_learning.py ../datasets/faces_train_128_10/filelist.txt -t scalar -i {str(alpha0)} -o {os.path.join(out_dir,str(alpha0))} -v'
os.system(cmd)