'''
Created on Oct 30, 2015

@author: kashefy
'''
from nose.tools import assert_equal
import os
import tempfile
import shutil
import numpy as np
import cv2 as cv2
import read_img as r

class TestReadImage:

    @classmethod
    def setup_class(self):
        
        self.dir_tmp = tempfile.mkdtemp()
        
        self.img1 = np.array([[[ 1,  2,  3],
                               [ 4,  5,  6]
                               ],
                              [[ 7,  8,  9],
                               [10, 11, 12]
                               ],
                              [[13, 14, 15],
                               [16, 17, 18],
                               ],
                              [[19, 20, 21],
                               [22, 23, 24]
                               ]
                              ])
        
        self.path_img1 = os.path.join(self.dir_tmp, "a.png")
        cv2.imwrite(self.path_img1, self.img1)
        
    @classmethod
    def teardown_class(self):
        
        shutil.rmtree(self.dir_tmp)
    
    def test_read_img_cv2_shape(self):
        
        img = r.read_img_cv2(self.path_img1)
        assert_equal(img.shape, (3, 4, 2))
        
    def test_read_img_cv2_pixels(self):
        
        img = r.read_img_cv2(self.path_img1)
        
        for ch in range(3):
            for row in range(4):
                for col in range(2):
                    assert_equal(img[ch][row][col], self.img1[row][col][ch])
        