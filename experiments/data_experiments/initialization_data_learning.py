import numpy as np
import os
from ast import literal_eval

lambda_parameters = np.array([1, 5, 10, 20, 40, 80, 160])
out_dir = 'initialization_data_parameter'
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
summary_table_dir = 'initialization_data_parameter/summary_table.csv'

with open(summary_table_dir, 'w+') as f:
    f.write('lambda0,fun,msg,nfev,njev,nregjev,nit,status,success,lambda_opt,l2_cost,psnr,ssim\n')
    for l in lambda_parameters:
        print(
            f'Executing the lambda initialization experiment with lambda:{l}')
        cmd = f'python ../../data_learning.py ../../datasets/cameraman_128_5/filelist.txt -t patch -prows 1 -pdata {str(l.tolist()).strip("[]").replace(",","")} -o {os.path.join(out_dir,str(l))} -v'
        os.system(cmd)
        ex_summary_path = os.path.join(out_dir, str(l), 'summary.out')
        ex_quality_path = os.path.join(out_dir, str(l), 'quality.out')

        info = [str(l)]
        s = open(ex_summary_path, 'r')
        lines = s.readlines()
        for line in lines:
            l = line.split(':')
            info.append(l[1].strip())

        qs = open(ex_quality_path, 'r').readlines()
        lambda_opt = qs[0].split(':')[1].strip()
        info.append(str(np.linalg.norm(np.array(literal_eval(lambda_opt)))))
        for q in qs[2:]:
            qus = q.split('\t')
            info.append(qus[2])
            info.append(qus[4])
            info.append(qus[6])
        f.write(','.join(info))
