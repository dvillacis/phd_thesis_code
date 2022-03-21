import unittest
from PIL import Image
import numpy as np
from bilearning.GPDenoising.gp_patch_denoising import gp_patch_denoise_ds
from bilearning.Operators.patch import Patch


class GPTVTest(unittest.TestCase):
    def test_ds_patch_denoise(self):
        n = 128
        data_parameter_gaussian = Patch(20.0 * np.ones(16),4,4)
        data_parameter_poisson = Patch(20.0 * np.ones(16),4,4)
        rec = gp_patch_denoise_ds(
            'datasets/cameraman_128_10/filelist.txt', data_parameter_gaussian, data_parameter_poisson, show=True)
        print(len(rec))
