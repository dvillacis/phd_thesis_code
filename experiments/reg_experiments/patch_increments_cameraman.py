
import numpy as np
import os
import sys
from ast import literal_eval
sys.path.append('../../')
from bilearning.Operators.patch import Patch

nrows = np.array([1,2,4,8,16,32])
alpha0 = 0.0155*np.ones(nrows[0]**2)

out_dir = 'patch_increments_cameraman'
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
        cmd = f'python ../../regularization_learning.py ../../datasets/cameraman_128_5/filelist.txt -t patch -t patch -prows {str(r)} -pdata {str(alpha0.tolist()).strip("[]").replace(",","")} -o {os.path.join(out_dir,str(r))} -v'
        print(cmd[:150])
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
            p = Patch(alpha_opt, r, r)
            alpha0 = p.map_to_img(np.ones((nrows[j], nrows[j])))
