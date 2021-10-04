import unittest
import numpy as np

from Learning.optimization import find_optimal_data_scalar

class OptimizationTest(unittest.TestCase):
    def test_data_learn(self):
        dsfile = 'datasets/cameraman_128_5/filelist.txt'
        # dsfile = 'datasets/faces_train_128_10/filelist.txt'
        initial_data_parameter = 1000.0
        optimal = find_optimal_data_scalar(dsfile,initial_data_parameter)
        print(optimal) 
