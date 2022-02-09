import os,sys
import numpy as np
from PIL import Image, ImageDraw

name = 'datasets/circles_128_10/circles_128_10'

w, h = 128, 128
shape = [(10, 10), (w - 10, h - 10)]
shape2 = [(30,30), (w - 30, h - 30)]
shape3 = [(50,50), (w - 50, h - 50)]
shape4 = [(70, 70), (w - 70, h - 70)]
shape5 = [(90, 90), (w - 90, h - 90)]

# Create circles image
original = Image.new('L',(w,h),color=255)
draw = ImageDraw.Draw(original)
draw.ellipse(shape,fill=0)
draw.ellipse(shape2, fill=255)
draw.ellipse(shape3, fill=0)
draw.ellipse(shape4, fill=255)
draw.ellipse(shape5, fill=0)
original.save(name+'_true_1.png')

# Adding gaussian noise
mean = 0
var = 0.2
mean2 = 0
var2 = 0.05

## Full image
noise = np.random.normal(mean, var**0.5, (w, h))
noisy_full = np.clip(np.array(original)/255 + noise,0.0,1.0)
noisy_full = Image.fromarray(noisy_full*255).convert('L')
noisy_full.save(name+'_data_1.png')

## Patch only
noise = np.random.normal(mean, var**0.5, (w//4, h//4))
noise = np.kron(np.eye(4),noise)
noisy_patch = np.clip(np.array(original)/255 + noise, 0.0, 1.0)
noisy_patch = Image.fromarray(noisy_patch*255).convert('L')
original.save(name+'_true_2.png')
noisy_patch.save(name+'_data_2.png')

## Two patch diff noise levels
noise = np.hstack(
    (np.random.normal(mean, var**0.5, (w//2, h//2)), np.zeros((w//2, h//2))))
noise_low = np.hstack(
    (np.zeros((w//2, h//2)),np.random.normal(mean2, var2**0.5, (w//2, h//2))))
noise = np.vstack((noise, noise_low))
noisy_patch_2 = np.clip(np.array(original)/255 + noise, 0.0, 1.0)
noisy_patch_2 = Image.fromarray(noisy_patch_2*255).convert('L')
original.save(name+'_true_3.png')
noisy_patch_2.save(name+'_data_3.png')

