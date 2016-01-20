'''
Created on Jan 20, 2016

@author: kashefy
'''
from nose.tools import assert_equal, assert_true, assert_list_equal,\
    assert_is_not_none
import os
import tempfile
import shutil
import numpy as np
import proto_utils as pu

import sys
CURRENT_MODULE_PATH = os.path.abspath(sys.modules[__name__].__file__)
TEST_DATA_DIRNAME = 'test_data'
TEST_NET_FILENAME = 'n1.prototxt'
        
class TestProtoUtils:

    @classmethod
    def setup_class(self):
        
        self.dir_tmp = tempfile.mkdtemp()
        
    @classmethod
    def teardown_class(self):
        
        shutil.rmtree(self.dir_tmp)
    
    def test_from_net_params_file(self):
        
        fpath = os.path.join(os.path.dirname(CURRENT_MODULE_PATH),
                             TEST_DATA_DIRNAME, TEST_NET_FILENAME)
        
        parser = pu.Parser()
        net_spec = parser.from_net_params_file(fpath)
        
        assert_is_not_none(net_spec)
        