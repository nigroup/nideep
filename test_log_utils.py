'''
Created on Oct 1, 2015

@author: kashefy
'''
import os
import sys
import shutil
import tempfile
from nose.tools import assert_equal, assert_false, \
    assert_is_instance, assert_true
import log_utils as lu

CURRENT_MODULE_PATH = os.path.abspath(sys.modules[__name__].__file__)
TEST_DATA_DIRNAME = 'test_data'
TEST_LOG_FILENAME = 'caffe.hostname.username.log.INFO.20150917-163712.31405'

class TestPID:

    def test_pid_from_str(self):
        
        n = 26943
        res = lu.pid_from_str('%s' % n)
        assert_is_instance(res, int)
        assert_equal(res, n)
        
    def test_pid_from_str_neg(self):
        
        n = -26943
        res = lu.pid_from_str('%s' % n)
        assert_is_instance(res, int)
        assert_equal(res, n)
        
    def test_pid_from_str_invalid_alpha(self):
        
        n = 26943
        x = ['a%s' % n, '%sa' % n, '269a43']
        
        for s in x:
        
            res = lu.pid_from_str(s)
            assert_is_instance(res, int)
            assert_equal(res, -1)
        
    def test_pid_from_logname(self):
        
        s = 'caffe.host.user.log.INFO.20151001-132750.26943'
        res = lu.pid_from_logname(s)
        assert_is_instance(res, int)
        assert_equal(res, 26943)
        
        n = 77775
        s = 'log.%s' % n
        res = lu.pid_from_logname(s)
        assert_is_instance(res, int)
        assert_equal(res, n)
        
    def test_pid_from_logname_invalid(self):
        
        s = 'caffe.host.user.log.INFO.20151001-132750.26943.txt'
        res = lu.pid_from_logname(s)
        assert_is_instance(res, int)
        assert_equal(res, -1)
        
class TestCaffeLog:
    
    @classmethod
    def setup_class(self):
        
        self.path_temp_dir = tempfile.mkdtemp()
        self.path_real_log = os.path.join(os.path.dirname(CURRENT_MODULE_PATH),
                                          TEST_DATA_DIRNAME,
                                          TEST_LOG_FILENAME)
        
    @classmethod
    def teardown_class(self):
        
        shutil.rmtree(self.path_temp_dir)
    
    def test_is_caffe_log(self):
        
        assert_true(lu.is_caffe_log(self.path_real_log))
        
    def test_is_caffe_log_invalid_prefix(self):
        
        fpath = os.path.join(self.path_temp_dir,
                             "foo.hostname.username.log.INFO.20150917-163712.31405")
        
        with open(fpath, 'w') as f:
            f.write('log file')
        
        assert_false(lu.is_caffe_log(fpath))
        
    def test_is_caffe_log_invalid_content(self):
        
        fpath = os.path.join(self.path_temp_dir,
                             "caffe.hostname.username.log.INFO.20150917-163712.31405")
        
        with open(fpath, 'w') as f:
            f.write('foo')
        
        assert_false(lu.is_caffe_log(fpath))
    