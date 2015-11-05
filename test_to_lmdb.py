'''
Created on Oct 30, 2015

@author: kashefy
'''
from nose.tools import assert_equal, assert_true
import os
import tempfile
import shutil
import numpy as np
import lmdb
import cv2 as cv2
import to_lmdb as tol

class TestImagesToLMDB:

    @classmethod
    def setup_class(self):
        
        self.dir_tmp = tempfile.mkdtemp()
        
        x = np.array([[[ 1,  2,  3],
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
        cv2.imwrite(self.path_img1, x)
        
    @classmethod
    def teardown_class(self):
        
        shutil.rmtree(self.dir_tmp)
    
    def test_img_str(self):
        
        s = '\x08\x03\x10\x04\x18\x02"\x18\x01\x04\x07\n\r\x10\x13\x16\x02\x05\x08\x0b\x0e\x11\x14\x17\x03\x06\t\x0c\x0f\x12\x15\x18(\x00'
        
        path_lmdb = os.path.join(self.dir_tmp, 'x_lmdb')
        tol.imgs_to_lmdb([self.path_img1], path_lmdb)
        assert_true(os.path.isdir(path_lmdb), "failed to save LMDB")
        
        env_src = lmdb.open(path_lmdb, readonly=True)
        
        count = 0
        
        with env_src.begin() as txn:
            
            cursor = txn.cursor()
            for key, value in cursor:
                
                assert_equal(key, '0000000000', "Unexpected key.")
                assert_equal(value, s, "Unexpected content.")
                
                count += 1
        
        assert_equal(count, 1, "Unexpected number of samples.")    
            
        
        