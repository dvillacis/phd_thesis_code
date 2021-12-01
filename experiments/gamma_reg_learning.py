import os
import numpy as np
from ast import literal_eval

gammas = np.array([5,10,100])
alpha0 = 0.0001
threshold_radius = 10.0

out_dir = 'gamma_reg_parameter'
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
summary_table_dir = os.path.join(out_dir,'summary_table.csv')

with open(summary_table_dir,'w+') as f:
    f.write('gamma,fun,msg,nfev,njev,nregjev,nit,status,success,alpha_opt,l2_cost,psnr,ssim\n')
    for gamma in gammas:
        print(f'Executing the gamma experiment with gamma:{gamma}')
        cmd = f'python ../regularization_learning.py ../datasets/cameraman_128_5/filelist.txt -t scalar -tr {str(threshold_radius)} -i {str(alpha0)} -g {str(gamma)} -o {os.path.join(out_dir,str(gamma))} -v'
        print(cmd)
        os.system(cmd)

        ex_summary_path = os.path.join(out_dir,str(gamma),'summary.out')
        ex_quality_path = os.path.join(out_dir,str(gamma),'quality.out')

        info = [str(gamma)]
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

    print(f'Executing the gamma experiment with no smoothing')
    threshold_radius = 1e-10
    cmd = f'python ../regularization_learning.py ../datasets/cameraman_128_5/filelist.txt -t scalar -tr {str(threshold_radius)} -i {str(alpha0)} -o {os.path.join(out_dir,"no_smoothing")} -v'
    print(cmd)
    os.system(cmd)

    ex_summary_path = os.path.join(out_dir,'no_smoothing','summary.out')
    ex_quality_path = os.path.join(out_dir,'no_smoothing','quality.out')

    info = [str(alpha0)]
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