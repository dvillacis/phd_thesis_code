import unittest
import numpy as np

from bilearning.Learning.optimization import find_optimal_data_scalar, find_optimal_data_patch

class OptimizationTest(unittest.TestCase):
    # def test_data_learn(self):
    #     dsfile = 'datasets/cameraman_128_5/filelist.txt'
    #     # dsfile = 'datasets/faces_train_128_10/filelist.txt'
    #     initial_data_parameter = 1000.0
    #     optimal = find_optimal_data_scalar(dsfile,initial_data_parameter)
    #     print(optimal) 
    
    def test_data_learn_patch(self):
        dsfile = 'datasets/cameraman_128_5/filelist.txt'
        # dsfile = 'datasets/faces_train_128_10/filelist.txt'
        initial_data_parameter = 10.0
        optimal = find_optimal_data_patch(dsfile,initial_data_parameter,show=True)
        print(optimal) 
