'''
Created on Oct 30, 2015

@author: kashefy
'''
from nose.tools import assert_equal, assert_false, \
    assert_list_equal, assert_true
from mock import patch, PropertyMock
import numpy as np
from numpy.testing import assert_array_equal
import inference as infr

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

class TestInference:
        
    @patch('inference.caffe.Net')
    def test_forward(self, mock_net):
        
        # fake minimal test data
        b = {k : Bunch(data=np.random.rand(3, 2)) for k in ['x', 'y', 'z']}
        
        # mock methods and properties of Net objects
        mock_net.return_value.forward.return_value = np.zeros(1)
        type(mock_net.return_value).blobs = PropertyMock(return_value=b)
        net = mock_net()
        
        assert_false(net.forward.called, "Problem with mocked forward()")
            
        out = infr.forward(net, ['x', 'z'])
        
        assert_true(net.forward.called, "Problem with mocked forward()")
        
        assert_list_equal(out.keys(), ['x', 'z'])
        for k in ['x', 'z']:
            assert_equal(out[k].shape, (3, 2),
                         msg="unexpected shape for blob %s" % k)
            
            assert_array_equal(b[k].data, out[k])
            
    