'''
Created on Jan 20, 2016

@author: kashefy
'''
from nose.tools import assert_is_not_none, assert_is_not, assert_equal
import os
from google.protobuf import text_format
from caffe.proto.caffe_pb2 import NetParameter
import proto_utils as pu

import sys
CURRENT_MODULE_PATH = os.path.abspath(sys.modules[__name__].__file__)
TEST_DATA_DIRNAME = 'test_data'
TEST_NET_FILENAME = 'n1.prototxt'
        
class TestProtoUtils:
    
    def test_from_net_params_file(self):
        
        fpath = os.path.join(os.path.dirname(CURRENT_MODULE_PATH),
                             TEST_DATA_DIRNAME, TEST_NET_FILENAME)
        
        parser = pu.Parser()
        net_spec = parser.from_net_params_file(fpath)
        
        assert_is_not_none(net_spec)
    
    def test_copy_msg(self):
        
        x = NetParameter()
        assert_is_not_none(x)
        y = pu.copy_msg(x, NetParameter)
        assert_is_not(x, y)
        assert_is_not_none(y)
      
class TestCopyNetParams:  
        
        def test_copy_net_params(self):
        
            fpath = os.path.join(os.path.dirname(CURRENT_MODULE_PATH),
                                 TEST_DATA_DIRNAME, TEST_NET_FILENAME)
            
            parser = pu.Parser()
            x = parser.from_net_params_file(fpath)
            
            y = pu.copy_net_params(x)
            assert_is_not(x, y, "References to the same instance.")
            assert_is_not_none(y)
            
            assert_equal(text_format.MessageToString(x), text_format.MessageToString(y))
            