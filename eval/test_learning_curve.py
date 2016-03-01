'''
Created on Oct 1, 2015

@author: kashefy
'''
import os
import sys
from nose.tools import assert_true,\
    assert_list_equal
from learning_curve import LearningCurve

CURRENT_MODULE_PATH = os.path.abspath(sys.modules[__name__].__file__)
ROOT_PKG_PATH = os.path.dirname(CURRENT_MODULE_PATH)
TEST_DATA_DIRNAME = 'test_data'
TEST_LOG_FILENAME = 'caffe.hostname.username.log.INFO.20150917-163712.31405'

class TestLearningCurve:
    
    @classmethod
    def setup_class(self):
        
        self.fpath = os.path.join(os.path.dirname(ROOT_PKG_PATH),
                                  TEST_DATA_DIRNAME,
                                  TEST_LOG_FILENAME)
        assert_true(os.path.isfile(self.fpath),
                    "Cannot test without log file.")
    
    def test_keys(self):
        
        lc = LearningCurve(self.fpath)
        train_keys, test_keys = lc.parse()
        assert_list_equal(train_keys, ['NumIters', 'Seconds', 'LearningRate', 'loss'])
        assert_list_equal(test_keys, ['NumIters', 'Seconds', 'LearningRate', 'accuracy', 'loss'])