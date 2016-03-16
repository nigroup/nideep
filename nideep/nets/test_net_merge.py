'''
Created on Jan 20, 2016

@author: kashefy
'''
from nose.tools import assert_equal, assert_true, assert_false,\
    assert_is_not_none, assert_is_instance, assert_greater, assert_list_equal,\
    assert_is_not
import os
import tempfile
import shutil
from google.protobuf import text_format
from caffe.proto.caffe_pb2 import NetParameter
from nideep.proto.proto_utils import Parser
import net_merge as mrg

import sys
CURRENT_MODULE_PATH = os.path.abspath(sys.modules[__name__].__file__)
ROOT_PKG_PATH = os.path.dirname(CURRENT_MODULE_PATH)
TEST_DATA_DIRNAME = 'test_data'
TEST_NET_FILENAME = 'n1.prototxt'

class TestNetMerge:

    @classmethod
    def setup_class(self):
        
        self.dir_tmp = tempfile.mkdtemp()
        
    @classmethod
    def teardown_class(self):
        
        shutil.rmtree(self.dir_tmp)
    
    def test_duplicate(self):
        
        fpath = os.path.join(os.path.dirname(ROOT_PKG_PATH),
                             TEST_DATA_DIRNAME, TEST_NET_FILENAME)
        
        n1 = Parser().from_net_params_file(fpath)
        n2 = Parser().from_net_params_file(fpath)
        
        n1_tmp = NetParameter(); n1_tmp.CopyFrom(n1)
        n2_tmp = NetParameter(); n2_tmp.CopyFrom(n2)
        s = mrg.merge_indep_net_spec([n1_tmp, n2_tmp])
        
        assert_is_not_none(s)
        assert_is_instance(s, str)
        assert_greater(len(s), 0)
        
        n = NetParameter()
        text_format.Merge(s, n)
        assert_is_not_none(n)
        
        # Data Layer from first network
        for l in n.layer:
            if l.type.lower() == 'data':
                for l1 in n1.layer:
                    if l1.type.lower() == 'data':
                        
                        dat_phase = [x.phase for x in l.include]
                        # compare test with test and train with train
                        if dat_phase == [x.phase for x in l1.include]:
                        
                            assert_is_not(l.top, l1.top)
                            assert_list_equal(list(l.top), list(l1.top))
                            assert_equal(l.data_param.source, l1.data_param.source)                        
                            assert_equal(l.data_param.backend, l1.data_param.backend)
                            assert_equal(l.data_param.batch_size, l1.data_param.batch_size)
                            assert_equal(l.transform_param.scale, l1.transform_param.scale)
        # For non-data layers
        
        # back up merged net
        for ni in [n1, n2]: 
            for l1 in ni.layer:
                found = False
                if l1.type.lower() != 'data':
                    
                    for l in n.layer:                    
                        if l.type.lower() == l1.type.lower() and \
                           [t.split('_nidx')[0] for t in l.top] == list(l1.top) and \
                           [b.split('_nidx')[0] for b in l.bottom] == list(l1.bottom):
                            
                            assert_true(l.name.startswith(l1.name))
                    
                            fnames1 = [f.name for f in l1.DESCRIPTOR.fields]
                            fnames = [f.name for f in l.DESCRIPTOR.fields]
                            assert_list_equal(fnames, fnames1)
                            
                            l.ClearField('name')
                            l.ClearField('top')
                            l.ClearField('bottom')
                            l1.ClearField('name')
                            l1.ClearField('top')
                            l1.ClearField('bottom')
                            
                            assert_equal(text_format.MessageToString(l), text_format.MessageToString(l1))
                            
                            found = True
                else:
                    continue # skip for data layers
                assert_true(found, "Failed to find %s in merged network!" % (l1.name,))
                
                
                
                
                
                
                
                
        