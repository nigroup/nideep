'''
Created on Oct 1, 2015

@author: kashefy
'''
import os
import sys
import shutil
import tempfile
from nose.tools import assert_equal, assert_false, assert_raises, \
    assert_is_instance, assert_is_none, assert_less, assert_true
import log_utils as lu

CURRENT_MODULE_PATH = os.path.abspath(sys.modules[__name__].__file__)
ROOT_PKG_PATH = os.path.dirname(CURRENT_MODULE_PATH)
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
        
    def test_read_pid(self):
        
        fpath = os.path.join(os.path.dirname(ROOT_PKG_PATH),
                             TEST_DATA_DIRNAME,
                             TEST_LOG_FILENAME)
        
        result = lu.read_pid(fpath)
        assert_is_instance(result, int)
        assert_equal(result, 31405)
        
    def test_read_pid_from_content(self):
        
        fpath = os.path.join(os.path.dirname(ROOT_PKG_PATH),
                             TEST_DATA_DIRNAME,
                             TEST_LOG_FILENAME)
        
        path_temp_dir = tempfile.mkdtemp()
        
        fpath2 = os.path.join(path_temp_dir, "foo.txt")
        
        shutil.copyfile(fpath, fpath2)
        
        try:
        
            assert_less(lu.pid_from_logname(fpath2), 0)
        
            result = lu.read_pid(fpath2)
            assert_is_instance(result, int)
            assert_equal(result, 31405)
            
        except Exception:
            pass
        shutil.rmtree(path_temp_dir)
            
    def test_read_pid_invalid(self):
        
        path_temp_dir = tempfile.mkdtemp()
        fpath = os.path.join(path_temp_dir, TEST_LOG_FILENAME)
        
        with open(fpath, 'w') as f:
            f.write('log file')
        
        assert_true(lu.is_caffe_info_log(fpath))
        
        fpath2 = os.path.join(path_temp_dir, "foo.txt")
        shutil.copyfile(fpath, fpath2)
        
        assert_raises(IOError, lu.read_pid, fpath2)
        
        shutil.rmtree(path_temp_dir)
        
class TestCaffeLog:
    
    @classmethod
    def setup_class(self):
        
        self.dir_tmp = tempfile.mkdtemp()
        self.path_real_log = os.path.join(os.path.dirname(ROOT_PKG_PATH),
                                          TEST_DATA_DIRNAME,
                                          TEST_LOG_FILENAME)
        
    @classmethod
    def teardown_class(self):
        
        shutil.rmtree(self.dir_tmp)
    
    def test_is_caffe_log_invalid_dir(self):
        
        assert_true(os.path.isdir(self.dir_tmp))
        assert_false(lu.is_caffe_log(self.dir_tmp))
        
    def test_is_caffe_log(self):
        
        assert_true(lu.is_caffe_log(self.path_real_log))
        
    def test_is_caffe_log_invalid_prefix(self):
        
        fpath = os.path.join(self.dir_tmp,
                             "foo.hostname.username.log.INFO.20150917-163712.31405")
        
        with open(fpath, 'w') as f:
            f.write('log file')
        
        assert_false(lu.is_caffe_log(fpath))
        
    def test_is_caffe_log_invalid_content(self):
        
        fpath = os.path.join(self.dir_tmp,
                             "caffe.hostname.username.log.INFO.20150917-163712.31405")
        
        with open(fpath, 'w') as f:
            f.write('foo')
        
        assert_false(lu.is_caffe_log(fpath))
        
    def test_is_caffe_info_log(self):
        
        assert_true(lu.is_caffe_log(self.path_real_log))
        
    def test_is_caffe_info_log_invalid_fname(self):
        
        fpath = os.path.join(self.dir_tmp,
                             "foo.hostname.username.log.ERROR.20150917-163712.31405")
        
        with open(fpath, 'w') as f:
            f.write('log file')
        
        assert_false(lu.is_caffe_info_log(fpath))
        
    def test_is_complete(self):
        
        assert_true(lu.is_complete(self.path_real_log))
                
        fpath = os.path.join(self.dir_tmp,
                             "foo.hostname.username.log.ERROR.20150917-163712.31405")
        
        with open(fpath, 'w') as f:
            f.write('log file')
        
        assert_false(lu.is_complete(fpath))
        
class TestReadMeUtils:
    
    @classmethod
    def setup_class(self):
        
        self.dir_tmp = tempfile.mkdtemp()
        self.path_real_log = os.path.join(os.path.dirname(ROOT_PKG_PATH),
                                          TEST_DATA_DIRNAME,
                                          TEST_LOG_FILENAME)
        shutil.copyfile(self.path_real_log,
                        os.path.join(self.dir_tmp, TEST_LOG_FILENAME))
        self.path_real_log = os.path.join(self.dir_tmp, TEST_LOG_FILENAME)
        self.path_readme = os.path.join(self.dir_tmp, 'readme.txt')
        with open(self.path_readme, 'w') as f:
            f.write('foo')
        
    @classmethod
    def teardown_class(self):
        
        shutil.rmtree(self.dir_tmp)
    
    def test_get_rel_readme_path_from_dir(self):
        
        p = lu.get_rel_readme_path(self.dir_tmp)
        assert_equal(p, self.path_readme)
        
    def test_get_rel_readme_path_from_log(self):
        
        p = lu.get_rel_readme_path(self.path_real_log)
        assert_equal(p, self.path_readme)
        
    def test_readme_to_str(self):
        
        assert_equal(lu.readme_to_str(self.path_readme), "foo")
        
class TestFindLine:
    
    @classmethod
    def setup_class(self):
        
        self.dir_tmp = tempfile.mkdtemp()
        
    @classmethod
    def teardown_class(self):
        
        shutil.rmtree(self.dir_tmp)
        
    def test_find_line(self):
        
        fpath = os.path.join(self.dir_tmp, "foo.txt")
        
        with open(fpath, 'w') as f:
            f.write('line one\n')
            f.write('line two\n')
            f.write('LINE x\n')
            f.write('line y\n')
            f.write('last line\n')
            
        assert_is_none(lu.find_line(fpath, 'hello'))
        assert_equal(lu.find_line(fpath, 'line'), 'line one' + os.linesep)
        assert_equal(lu.find_line(fpath, 'LINE'), 'LINE x' + os.linesep)
        
        