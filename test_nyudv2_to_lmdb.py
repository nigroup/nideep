from nose.tools import assert_is_instance, assert_list_equal, assert_raises, \
    assert_true
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
        