import os

from numpy import dtype

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