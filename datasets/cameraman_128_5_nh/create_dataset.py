from json import load
import os,sys
import numpy as np
from PIL import Image, ImageDraw


# Adding gaussian noise
mean = 0
var = 0.0025
mean2 = 0
var2 = 0.04

## Patch only
with Image.open('cameraman_128_5_data_2.png') as original:
    noise = np.random.normal(mean2, var2**0.5, (32, 32))
    #noise_full = np.random.normal(mean, var**0.5, (32, 32))
    noise = np.kron(
        np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]), noise)
    # noise_full = np.kron(
    #     np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]]), noise_full)
    noisy_patch = np.clip(np.array(original)/255 + noise, 0.0, 1.0)
    noisy_patch = Image.fromarray(noisy_patch*255).convert('L')
    noisy_patch.save('cameraman_128_5_data_1.png')


