'''
Created on Oct 30, 2015

@author: kashefy
'''
from nose.tools import assert_equal, assert_false, \
    assert_list_equal, assert_true
from mock import patch, PropertyMock
import os
import tempfile
import shutil
import numpy as np
from numpy.testing import assert_array_equal
import h5py
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
            
        # repeat with smaller set of keys
        out = infr.forward(net, ['z'])
        assert_list_equal(out.keys(), ['z'])
        assert_equal(out['z'].shape, (3, 2), msg="unexpected shape for blob z")
        assert_array_equal(b['z'].data, out['z'])

class TestHDF5Inference:
    
    @classmethod
    def setup_class(self):
        
        self.dir_tmp = tempfile.mkdtemp()
        
    @classmethod
    def teardown_class(self):
        
        shutil.rmtree(self.dir_tmp)
    
    @patch('inference.caffe.Net')
    def test_infer_to_h5_fixed_dims(self, mock_net):
        
        # fake minimal test data
        b = {k : Bunch(data=np.random.rand(1, 1, 3, 2)) for k in ['x', 'y', 'z']}
        
        # mock methods and properties of Net objects
        mock_net.return_value.forward.return_value = np.zeros(1)
        type(mock_net.return_value).blobs = PropertyMock(return_value=b)
        net = mock_net()
        
        fpath = os.path.join(self.dir_tmp, 'test_infer_to_h5_fixed_dims.h5')
        assert_false(os.path.isfile(fpath))
        
        out = infr.infer_to_h5_fixed_dims(net, ['x', 'z'], 1, fpath)
        assert_true(os.path.isfile(fpath))
        assert_equal(net.forward.call_count, 1)
        
        # check db content
        with h5py.File(fpath, "r") as f:
            assert_list_equal([str(k) for k in f.keys()], ['x', 'z'])
            
            for k in ['x', 'z']:
                assert_equal(f[k].shape, (1, 1, 3, 2),
                             msg="unexpected shape for blob %s" % k)
            assert_array_equal(b[k].data, f[k])
        
        assert_list_equal(out, [1, 1])
            
    