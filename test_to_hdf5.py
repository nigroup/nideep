'''
Created on Jan 20, 2016

@author: kashefy
'''
from nose.tools import assert_equal, assert_true, assert_list_equal
import os
import tempfile
import shutil
import numpy as np
import h5py
import to_hdf5 as to
        
class TestArraysToHDF5:

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
        
        self.arr = [x, x+1]
        
    @classmethod
    def teardown_class(self):
        
        shutil.rmtree(self.dir_tmp)
    
    def test_arr_single(self):
        
        # use the module and test it
        fpath = os.path.join(self.dir_tmp, 'xarr1.h5')
        to.arrays_to_h5_fixed([self.arr[0]], 'x', fpath)
        
        with h5py.File(fpath, 'r') as h:
            assert_list_equal(h.keys(), ['x'])
            assert_equal(1, len(h['x']))  
            assert_true(np.all(self.arr[0]==h['x'][:]))
    
    def test_arr(self):
        
        fpath = os.path.join(self.dir_tmp, 'xarr.h5')
        to.arrays_to_h5_fixed(self.arr, 'x', fpath)

        with h5py.File(fpath, 'r') as h:
            assert_list_equal(h.keys(), ['x'])
            assert_equal(2, len(h['x']))
            for x, y in zip(self.arr, h['x'][:]):
                assert_true(np.all(x==y))
        