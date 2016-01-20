'''
Created on Jan 20, 2016

@author: kashefy
'''
from nose.tools import assert_equal, assert_true, assert_list_equal
import os
import tempfile
import shutil
import numpy as np
import proto_utils as pu
        
class TestProtoUtils:

    @classmethod
    def setup_class(self):
        
        self.dir_tmp = tempfile.mkdtemp()
        
    @classmethod
    def teardown_class(self):
        
        shutil.rmtree(self.dir_tmp)
    
    def test_net_merge(self):
        
        assert_true(True)