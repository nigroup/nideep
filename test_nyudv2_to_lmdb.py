from nose.tools import assert_is_instance, assert_list_equal, assert_raises
import os
import tempfile
import shutil
import numpy as np
from scipy import io
import nyudv2_to_lmdb as n2l
    
class TestHandlingSplitsFile:
    
    @classmethod
    def setup_class(self):
        
        self.path_temp_dir = tempfile.mkdtemp()
        self.path_splits = os.path.join(self.path_temp_dir, 'foo.mat')
        
        data = {'testNdxs': np.array([[2], [4], [10]])}
        io.savemat(self.path_splits, data, oned_as='column')
        
    @classmethod
    def teardown_class(self):
        
        shutil.rmtree(self.path_temp_dir)
        
    def test_invalid_path_dir(self):
        
        assert_raises(IOError, n2l.split_matfile_to_val_list, os.curdir)
        
    def test_invalid_path(self):
        
        assert_raises(IOError, n2l.split_matfile_to_val_list, '/foo/bar.mat')
        
    def test_val_list(self):
        
        val_list = n2l.split_matfile_to_val_list(self.path_splits)
        assert_is_instance(val_list, list)
        assert_list_equal(val_list, [[2], [4], [10]])
        