from nose.tools import assert_raises
import os
import tempfile
import shutil
import nyudv2_to_lmdb as n2l
    
class TestHandlingSplitsFile:
    
    @classmethod
    def setup_class(self):
        
        self.path_temp_dir = tempfile.mkdtemp()
        
    @classmethod
    def teardown_class(self):
        
        shutil.rmtree(self.path_temp_dir)
        
    def test_invalid_path_dir(self):
        
        assert_raises(IOError, n2l.split_matfile_to_val_list, os.curdir)
        
    def test_invalid_path(self):
        
        assert_raises(IOError, n2l.split_matfile_to_val_list, '/foo/bar.mat')