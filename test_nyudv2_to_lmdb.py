from nose.tools import assert_is_instance, assert_list_equal, assert_raises, \
    assert_true, assert_equal
import os
import tempfile
import shutil
import numpy as np
from scipy import io
import nyudv2_to_lmdb as n2l
    
class TestHandlingSplitsFile:
    
    @classmethod
    def setup_class(self):
        
        self.dir_tmp = tempfile.mkdtemp()
        self.path_splits = os.path.join(self.dir_tmp, 'foo.mat')
        
        data = {'testNdxs': np.array([[2], [4], [10]])}
        io.savemat(self.path_splits, data, oned_as='column')
        
        self.path_other = os.path.join(self.dir_tmp, 'bar.mat')
        data = {'foo': np.array([[2], [4]])}
        io.savemat(self.path_other, data, oned_as='column')
        
    @classmethod
    def teardown_class(self):
        
        shutil.rmtree(self.dir_tmp)
        
    def test_invalid_path_dir(self):
        
        assert_raises(IOError, n2l.split_matfile_to_val_list, os.curdir)
        
    def test_invalid_path(self):
        
        assert_raises(IOError, n2l.split_matfile_to_val_list, '/foo/bar.mat')
        
    def test_invalid_ext(self):
        
        fpath = os.path.join(self.dir_tmp, 'foo.txt')
        with open(fpath, 'w') as f:
            f.write('hello')
        
        assert_true(os.path.isfile(fpath))
        assert_raises(IOError, n2l.split_matfile_to_val_list, fpath)
        
    def test_val_list(self):
        
        val_list = n2l.split_matfile_to_val_list(self.path_splits)
        assert_is_instance(val_list, list)
        assert_list_equal(val_list, [1, 3, 9])
        
    def test_val_list_other(self):
        
        assert_raises(KeyError, n2l.split_matfile_to_val_list, self.path_other)
        
class TestBigArrToArrs:
    
    def test_big_arr_to_arrs_single(self):
        
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
        y = np.expand_dims(x, axis=0)
        z = n2l.big_arr_to_arrs(y)
        
        assert_is_instance(z, list)
        assert_equal(len(z), 1)
        for i in range(3):
            for j in range(4):
                for k in range(2):
                    assert_equal(z[0][j][i][k], x[j][k][i])
        
        
        