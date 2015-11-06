'''
Created on Oct 30, 2015

@author: kashefy
'''
from nose.tools import assert_equal, assert_almost_equals
from mock import patch
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
    
    def test_read_img_cv2_dtype(self):
        
        img = r.read_img_cv2(self.path_img1)
        assert_equal(img.dtype, np.dtype('uint8'))
        
    def test_read_img_cv2_subtract_mean(self):
        
        m = np.array((1., 2. ,3.))
        img = r.read_img_cv2(self.path_img1, mean=m)

        for ch in range(3):
            for row in range(4):
                for col in range(2):
                    assert_equal(img[ch][row][col], self.img1[row][col][ch]-m[ch])
    
    def test_read_img_PIL_shape(self):
        
        assert_equal(r.read_img_PIL(self.path_img1).shape, (3, 4, 2))
        
    def test_read_img_PIL_pixels(self):
        
        img = r.read_img_PIL(self.path_img1)
        
        for ch in range(3):
            for row in range(4):
                for col in range(2):
                    assert_equal(img[ch][row][col], self.img1[row][col][ch])
        
    def test_read_img_PIL_subtract_mean(self):
        
        m = np.array((1., 2. ,3.))
        img = r.read_img_PIL(self.path_img1, mean=m)

        for ch in range(3):
            for row in range(4):
                for col in range(2):
                    assert_equal(img[ch][row][col], self.img1[row][col][ch]-m[ch])
                 
    @patch('read_img.caffe')   
    def test_read_img_caf_shape(self, mock_caffe):
        
        mock_caffe.io.load_image.return_value = np.array([[[0.01176471,  0.00784314,  0.00392157],
                                                           [0.02352941,  0.01960784,  0.01568628]
                                                           ],
                                                          [[0.03529412,  0.03137255,  0.02745098],
                                                            [0.04705882, 0.04313726, 0.03921569],
                                                            ],
                                                          [[0.05882353,  0.05490196,  0.05098039],
                                                           [0.07058824,  0.06666667,  0.0627451 ]
                                                           ],
                                                          [[0.08235294,  0.07843138,  0.07450981],
                                                           [0.09411765,  0.09019608,  0.08627451]
                                                           ]
                                                          ])
        assert_equal(r.read_img_caf(self.path_img1).shape, (3, 4, 2))
                    
    @patch('read_img.caffe')   
    def test_read_img_caf_pixels(self, mock_caffe):
        
        mock_caffe.io.load_image.return_value = np.array([[[0.01176471,  0.00784314,  0.00392157],
                                                           [0.02352941,  0.01960784,  0.01568628]
                                                           ],
                                                          [[0.03529412,  0.03137255,  0.02745098],
                                                            [0.04705882, 0.04313726, 0.03921569],
                                                            ],
                                                          [[0.05882353,  0.05490196,  0.05098039],
                                                           [0.07058824,  0.06666667,  0.0627451 ]
                                                           ],
                                                          [[0.08235294,  0.07843138,  0.07450981],
                                                           [0.09411765,  0.09019608,  0.08627451]
                                                           ]
                                                          ])
        
        img = r.read_img_caf(self.path_img1)
        
        for ch in range(3):
            for row in range(4):
                for col in range(2):
                    assert_almost_equals(img[ch][row][col], self.img1[row][col][ch], places=5)
                    
    @patch('read_img.caffe')   
    def test_read_img_caf_subtract_mean(self, mock_caffe):
        
        mock_caffe.io.load_image.return_value = np.array([[[0.01176471,  0.00784314,  0.00392157],
                                                           [0.02352941,  0.01960784,  0.01568628]
                                                           ],
                                                          [[0.03529412,  0.03137255,  0.02745098],
                                                            [0.04705882, 0.04313726, 0.03921569],
                                                            ],
                                                          [[0.05882353,  0.05490196,  0.05098039],
                                                           [0.07058824,  0.06666667,  0.0627451 ]
                                                           ],
                                                          [[0.08235294,  0.07843138,  0.07450981],
                                                           [0.09411765,  0.09019608,  0.08627451]
                                                           ]
                                                          ])
        
        m = np.array((1., 2. ,3.))
        img = r.read_img_caf(self.path_img1, mean=m)
        
        for ch in range(3):
            for row in range(4):
                for col in range(2):
                    assert_almost_equals(img[ch][row][col], self.img1[row][col][ch]-m[ch], places=5)
    