import sys
import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from bilearning.Learning.cost import l2_cost
from bilearning.Operators.patch import patch

def write_scalar_report(report_dir,optimal,optimal_ds):
    with open(os.path.join(report_dir,'summary.out'),'w') as s:
        s.write(f'fun:{optimal.fun}\nmessage:{optimal.message}\nnfev:{optimal.nfev}\nnjev:{optimal.njev}\nn_reg_jev:{optimal.n_reg_jev}\nnit:{optimal.nit}\nstatus:{optimal.status}\nsuccess:{optimal.status}')

    with open(os.path.join(report_dir,'quality.out'),'w') as q:
        q.write(f'optimal data parameter: {optimal.x}\n')
        q.write('img_num\tl2_noisy\tl2_rec\tpsnr_noisy\tpsnr_rec\tssim_noisy\tssim_rec\n')
        img_num=0
        for k in optimal_ds.keys():
            img_num += 1
            original = optimal_ds[k][0]
            noisy = optimal_ds[k][1]
            rec = optimal_ds[k][2]
            l2_noisy = l2_cost(original,noisy)
            l2_rec = l2_cost(original,rec)
            psnr_noisy = psnr(original,noisy)
            psnr_rec = psnr(original,rec)
            ssim_noisy = ssim(original,noisy)
            ssim_rec = ssim(original,rec)
            q.write(f'{img_num}\t{l2_noisy}\t{l2_rec}\t{psnr_noisy}\t{psnr_rec}\t{ssim_noisy}\t{ssim_rec}\n')
            Image.fromarray(original*255).convert('L').save(os.path.join(report_dir,f'original_{img_num}.png'))
            Image.fromarray(noisy*255).convert('L').save(os.path.join(report_dir,f'noisy_{img_num}.png'))
            Image.fromarray(rec*255).convert('L').save(os.path.join(report_dir,f'rec_{img_num}.png'))

def write_patch_report(report_dir,optimal,optimal_ds):
    with open(os.path.join(report_dir,'summary.out'),'w') as s:
        s.write(f'fun:{optimal.fun}\nmessage:{optimal.message}\nnfev:{optimal.nfev}\nnjev:{optimal.njev}\nn_reg_jev:{optimal.n_reg_jev}\nnit:{optimal.nit}\nstatus:{optimal.status}\nsuccess:{optimal.status}')
        #s.write(f'{optimal}')
        
    with open(os.path.join(report_dir,'quality.out'),'w') as q:
        q.write(f'optimal data parameter: {optimal.x.data.tolist()}\n')
        q.write('img_num\tl2_noisy\tl2_rec\tpsnr_noisy\tpsnr_rec\tssim_noisy\tssim_rec\n')
        img_num=0
        for k in optimal_ds.keys():
            img_num += 1
            original = optimal_ds[k][0]
            noisy = optimal_ds[k][1]
            rec = optimal_ds[k][2]
            x_img = optimal.x.map_to_img(original).reshape(original.shape)
            x_img = (x_img.reshape(original.shape))/np.max(x_img)
            l2_noisy = l2_cost(original,noisy)
            l2_rec = l2_cost(original,rec)
            psnr_noisy = psnr(original,noisy)
            psnr_rec = psnr(original,rec)
            ssim_noisy = ssim(original,noisy)
            ssim_rec = ssim(original,rec)
            q.write(f'{img_num}\t{l2_noisy}\t{l2_rec}\t{psnr_noisy}\t{psnr_rec}\t{ssim_noisy}\t{ssim_rec}\n')
            Image.fromarray(original*255).convert('L').save(os.path.join(report_dir,f'original_{img_num}.png'))
            Image.fromarray(noisy*255).convert('L').save(os.path.join(report_dir,f'noisy_{img_num}.png'))
            Image.fromarray(rec*255).convert('L').save(os.path.join(report_dir,f'rec_{img_num}.png'))
            if len(x_img)>1:
                Image.fromarray(((x_img-x_img.min())/(x_img.max()-x_img.min()))*255).convert('L').save(os.path.join(report_dir,f'par_{img_num}.png'))


