import numpy as np
import os
import sys
from ast import literal_eval

def patch(x,out):
    nx,ny = out.shape
    px = int(np.sqrt(len(x)))
    x = x.reshape((px,px))
    x = np.kron(x,np.ones((nx//px,ny//px)))
    return x.ravel()

nrows = np.array([4,8,16,32])
alpha0 = 0.0155*np.ones(16)

out_dir = 'cameraman_patch_increments_reg_5'
#out_dir = os.path.join(out_dir,str(patch_size))
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
summary_table_dir = os.path.join(out_dir,'summary_table.csv')
with open(summary_table_dir,'w+') as f:
    f.write('patch_size,fun,msg,nfev,njev,nregjev,nit,status,success,alpha_opt,l2_cost,psnr,ssim\n')
    j=0
    for r in nrows:
        j+=1
        print(f'Executing the patch increment experiment with patch size:{r}x{r}')
        cmd = f'python ../regularization_learning.py ../datasets/cameraman_128_5/filelist.txt -t patch -t patch -prows {str(r)} -pdata {str(alpha0.tolist()).strip("[]").replace(",","")} -o {os.path.join(out_dir,str(r))} -v'
        print(cmd[:100])
        os.system(cmd)
        ex_summary_path = os.path.join(out_dir,str(r),'summary.out')
        ex_quality_path = os.path.join(out_dir,str(r),'quality.out')
        
        info = [str(r)]
        s = open(ex_summary_path,'r')
        lines = s.readlines()
        for line in lines:
            l = line.split(':')
            info.append(l[1].strip())

        qs = open(ex_quality_path,'r').readlines()
        alpha_opt_le = literal_eval(qs[0].split(':')[1].strip())
        alpha_opt = np.array(alpha_opt_le)
        print(alpha_opt.shape)
        #print(alpha_opt,type(alpha_opt),np.linalg.norm(alpha_opt))
        info.append(str(np.linalg.norm(alpha_opt)))
        for q in qs[2:]:
            qus = q.split('\t')
            info.append(qus[2])
            info.append(qus[4])
            info.append(qus[6])
        f.write(','.join(info))

        if j < len(nrows):
            alpha0 = patch(alpha_opt,np.ones((nrows[j],nrows[j])))