import unittest
import numpy as np
from bilearning.Dataset.load_dataset import load_ds_file

class DatasetTest(unittest.TestCase):
    def test_load(self):
        dsfile = 'datasets/cameraman_128_5/filelist.txt'
        ds = load_ds_file(dsfile)
        print(ds)