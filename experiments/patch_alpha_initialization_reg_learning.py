import numpy as np
import os
import sys
from ast import literal_eval
np.set_printoptions(threshold=sys.maxsize)

patch_size = 2
alpha_parameters = np.array([0.00001,0.0001,0.001,0.01,0.1,0.2,0.3])
out_dir = 'patch_alpha_initialization_reg_parameter'
out_dir = os.path.join(out_dir,str(patch_size))
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
summary_table_dir = os.path.join(out_dir,'summary_table.csv')

with open(summary_table_dir,'w+') as f:
    f.write('alpha0,fun,msg,nfev,njev,nregjev,nit,status,success,alpha_opt,l2_cost,psnr,ssim\n')
    for alpha in alpha_parameters:
        print(f'Executing the alpha initialization experiment with alpha:{alpha}')
        cmd = f'python ../regularization_learning.py ../datasets/cameraman_128_5/filelist.txt -t patch -ps {str(patch_size)} -i {str(alpha)} -o {os.path.join(out_dir,str(alpha))} -v'
        os.system(cmd)
        ex_summary_path = os.path.join(out_dir,str(alpha),'summary.out')
        ex_quality_path = os.path.join(out_dir,str(alpha),'quality.out')
        

        info = [str(alpha)]
        s = open(ex_summary_path,'r')
        lines = s.readlines()
        for line in lines:
            l = line.split(':')
            info.append(l[1].strip())

        qs = open(ex_quality_path,'r').readlines()
        alpha_opt_le = literal_eval(qs[0].split(':')[1].strip())
        alpha_opt = np.array(alpha_opt_le)
        #print(alpha_opt,type(alpha_opt),np.linalg.norm(alpha_opt))
        info.append(str(np.linalg.norm(alpha_opt)))
        for q in qs[2:]:
            qus = q.split('\t')
            info.append(qus[2])
            info.append(qus[4])
            info.append(qus[6])
        f.write(','.join(info))

