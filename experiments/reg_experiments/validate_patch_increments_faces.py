from json import load
import numpy as np
import sys, os
from PIL import Image

sys.path.append('../../')

from bilearning.TVDenoising.patch_denoising import patch_denoise
from bilearning.TVDenoising.scalar_denoising import denoise
from bilearning.Operators.patch import Patch, OnesPatch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from bilearning.Learning.cost import l2_cost
from bilearning.Dataset.load_dataset import get_image_pair_by_key,load_ds_file

patch_increments_dir = 'patch_increments_faces'

patch_sizes = [1,2,4,8,16,32]

ds_file = '../../datasets/faces_val_128_10/filelist.txt'

def quality_patch(original,noisy,patch_size):
    exp_dir = os.path.join(patch_increments_dir, str(patch_size), 'quality.out')
    
    if patch_size == 1:
        reconstruction = denoise(noisy, 14.138499, 1, niter=3000)
    else:
        with open(exp_dir, 'r') as quality_file:
            par = quality_file.readline().split(': ')[1][1:-2]
            par = np.fromstring(par, dtype=float, sep=', ')
            par = Patch(par, patch_size, patch_size)
            #print(par)
            reconstruction = patch_denoise(noisy, data_parameter=OnesPatch(patch_size, patch_size), reg_parameter=par, niter=3000)
    l2_noisy = l2_cost(original, noisy)
    l2_rec = l2_cost(original, reconstruction)
    psnr_noisy = psnr(original, noisy)
    psnr_rec = psnr(original, reconstruction)
    #ssim_noisy = ssim(original, noisy)
    ssim_rec = ssim(original, reconstruction)
    return reconstruction,ssim_rec


with open(os.path.join(patch_increments_dir, 'validation/validation.out'), 'w') as out:
    out.write('img_num\tnoisy\tscalar\t2x2\t4x4\t8x8\t16x16\t32x32\n')
    ds = load_ds_file(ds_file)
    img_num = 0
    for k in ds.keys():
        print(k,ds[k])
        img_num += 1
        out.write(f'{img_num}\t')
        original,noisy = get_image_pair_by_key(ds_file,k)
        ssim_noisy = ssim(original, noisy)
        out.write(f'{ssim_noisy}\t')
        for p in patch_sizes:
            rec,ssim_p = quality_patch(original, noisy, p)
            print(f'p:{p} ssim:{ssim_p}')
            out.write(f'{ssim_p}\t')
            Image.fromarray(rec*255).convert('L').save(os.path.join(patch_increments_dir,f'validation/{p}',f'rec_{img_num}.png'))
        out.write('\n')
                