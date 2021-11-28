import os
import numpy as np
from PIL import Image

def open_image(path):
    img = np.array(Image.open(path))
    img = img/np.max(img)
    return img

def load_ds_file(dataset_file_path):
    if not os.path.isfile(dataset_file_path):
        raise ValueError('Dataset file not found')
    dsdir = os.path.dirname(dataset_file_path)
    dataset = {}
    with open(dataset_file_path,'r') as f:
        for line in f:
            orig_path,noisy_path = line.split(',')
            dataset.update({os.path.join(dsdir,orig_path):os.path.join(dsdir,noisy_path)})
    return dataset

def get_image_pair(dataset_file_path,index):
    ds = load_ds_file(dataset_file_path)
    i = 0
    if i > len(ds):
        raise ValueError('Index out of bounds...')
    for k in ds.keys():
        if i == index:
            orig_path = k
            noisy_path = ds[k]
            orig = open_image(orig_path)
            noisy = open_image(noisy_path)
            return (orig,noisy)

