'''
Created on Oct 30, 2015

@author: kashefy
'''
from nose.tools import assert_false, assert_true, assert_is_instance,\
    assert_equal, assert_greater, assert_in
import os
import tempfile
import shutil
import sys
from iow import file_system_utils as fs
from eval.log_utils import is_caffe_info_log

CURRENT_MODULE_PATH = os.path.abspath(sys.modules[__name__].__file__)
ROOT_PKG_PATH = os.path.dirname(CURRENT_MODULE_PATH)
TEST_DATA_DIRNAME = 'test_data'

class TestFSUtils:
    
    @classmethod
    def setup_class(self):
        
        self.dir_ = os.path.join(os.path.dirname(ROOT_PKG_PATH),
                                          TEST_DATA_DIRNAME)
        
    def test_filter_is_img(self):
        
        assert_false(fs.filter_is_img('foo.bar'))
        assert_false(fs.filter_is_img('foo.png.bar'))
        assert_true(fs.filter_is_img('foo.png'))
        assert_true(fs.filter_is_img('foo.jpg'))
        
    def test_gen_paths_no_filter(self):
        
        flist = fs.gen_paths(self.dir_)
        assert_is_instance(flist, list)
        assert_greater(len(flist), 0)
        
    def test_gen_paths_is_caffe_log(self):
        
        flist = fs.gen_paths(self.dir_, is_caffe_info_log)
        assert_is_instance(flist, list)
        assert_equal(len(flist), 1)
        assert_true('.log.' in flist[0] and '.INFO.' in flist[0])
        
    def test_gen_paths_no_imgs_found(self):
        
        flist = fs.gen_paths(self.dir_, fs.filter_is_img)
        assert_equal(len(flist), 0)
        
class FNamePairsTest:
    
    @classmethod
    def setup_class(self):
        
        self.dir_tmp = tempfile.mkdtemp()
        self.dir_ = os.path.join(os.path.dirname(ROOT_PKG_PATH),
                                          TEST_DATA_DIRNAME)
        shutil.copytree(self.dir_, self.dir_tmp)
        
    @classmethod
    def teardown_class(self):
        
        shutil.rmtree(self.dir_tmp)
        
    def test_fname_pairs(self):
    
        a = fs.gen_paths(self.dir_, is_caffe_info_log)
        b = fs.gen_paths(self.dir_tmp, is_caffe_info_log)
        pairs = fs.fname_pairs(a, b)
        
        for x, y in pairs:
            assert_in(x, a)
            assert_in(y, b)
        