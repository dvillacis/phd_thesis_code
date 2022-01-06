import unittest
import numpy as np
from bilearning.Operators.patch import Patch


class PatchTest(unittest.TestCase):
    def test_patch(self):
        np.random.seed(12345)
        p = Patch(np.array([1,2,3,4]),2,2)
        img = np.ones((10,10))
        p_img = p.map_to_img(img)
        print(p_img.reshape(img.shape))
        red_p = p.reduce_from_img(p_img.reshape(img.shape))
        print(red_p.reshape((p.px,p.py)))
    
    def test_patch2(self):
        np.random.seed(12345)
        p = Patch(np.array([1, 2, 3, 4]), 1, 4)
        img = np.ones((12, 12))
        p_img = p.map_to_img(img)
        print(p_img.reshape(img.shape))
        red_p = p.reduce_from_img(p_img.reshape(img.shape))
        print(red_p)
        print(red_p.reshape((p.px, p.py)))

    def test_patch3(self):
        np.random.seed(12345)
        p = Patch(np.array([1, 2, 3, 4]), 4, 1)
        img = np.ones((12, 12))
        p_img = p.map_to_img(img)
        print(p_img.reshape(img.shape))
        red_p = p.reduce_from_img(p_img.reshape(img.shape))
        print(red_p)
        print(red_p.reshape((p.px, p.py)))

    def test_patch4(self):
        np.random.seed(12345)
        p = Patch(np.array([1, 2]), 1, 2)
        img = np.ones((12, 12))
        p_img = p.map_to_img(img)
        print(p_img.reshape(img.shape))
        red_p = p.reduce_from_img(p_img.reshape(img.shape))
        print(red_p)
        print(red_p.reshape((p.px, p.py)))

    def test_patch5(self):
        np.random.seed(12345)
        p = Patch(np.array([1, 2]), 2, 1)
        img = np.ones((12, 12))
        p_img = p.map_to_img(img)
        print(p_img.reshape(img.shape))
        red_p = p.reduce_from_img(p_img.reshape(img.shape))
        print(red_p)
        print(red_p.reshape((p.px, p.py)))

    def test_patch5(self):
        np.random.seed(12345)
        p = Patch(np.array([1]), 1, 1)
        img = np.ones((12, 12))
        p_img = p.map_to_img(img)
        print(p_img.reshape(img.shape))
        red_p = p.reduce_from_img(p_img.reshape(img.shape))
        print(red_p)
        print(red_p.reshape((p.px, p.py)))
