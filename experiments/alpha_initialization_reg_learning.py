import numpy as np
import os
from ast import literal_eval

alpha_parameters = np.array([0.00001,0.0001,0.001,0.01,0.1,0.2,0.3])
out_dir = 'alpha_initialization_reg_parameter'
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
summary_table_dir = 'alpha_initialization_reg_parameter/summary_table.csv'

with open(summary_table_dir,'w+') as f:
    f.write('alpha0,fun,msg,nfev,njev,nregjev,nit,status,success,alpha_opt,l2_cost,psnr,ssim\n')
    for alpha in alpha_parameters:
        print(f'Executing the alpha initialization experiment with alpha:{alpha}')
        cmd = f'python ../regularization_learning.py ../datasets/cameraman_128_5/filelist.txt -t scalar -i {str(alpha)} -o {os.path.join(out_dir,str(alpha))} -v'
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
        alpha_opt = qs[0].split(':')[1].strip()
        info.append(str(np.linalg.norm(np.array(literal_eval(alpha_opt)))))
        for q in qs[2:]:
            qus = q.split('\t')
            info.append(qus[2])
            info.append(qus[4])
            info.append(qus[6])
        f.write(','.join(info))

