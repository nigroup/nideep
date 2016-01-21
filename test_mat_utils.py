'''
Created on Oct 28, 2015

@author: kashefy
'''
from nose.tools import assert_equals, assert_true, assert_raises
import numpy as np
import mat_utils as mu

class TestTranspose:

    def test_cwh_to_chw_invalid_dims(self):
        
        assert_raises(AttributeError, mu.cwh_to_chw, np.random.rand(3))
        assert_raises(AttributeError, mu.cwh_to_chw, np.random.rand(3, 2))
        assert_raises(AttributeError, mu.cwh_to_chw, np.random.rand(5, 4, 3, 2))
        
    def test_cwh_to_chw(self):
        
        x = np.array([[[ 1,  2,  3],
                       [ 4,  5,  6]
                       ],
                      [[ 7,  8,  9],
                       [10, 11, 12]
                       ],
                      [[13, 14, 15],
                       [16, 17, 18],
                       ],
                      [[19, 20, 21],
                       [22, 23, 24]
                       ]
                      ])
        
        assert_equals(x.shape[0], 4) # c
        assert_equals(x.shape[1], 2) # w
        assert_equals(x.shape[2], 3) # h
        
        y = mu.cwh_to_chw(x)
        
        assert_equals(y.shape[0], 4)
        assert_equals(y.shape[1], 3)
        assert_equals(y.shape[2], 2)
        
        assert_equals(x[3][1][2], y[3][2][1])
        
        for i in range(4):
            for j in range(2):
                for k in range(3):
                    assert_equals(x[i][j][k], y[i][k][j])
        
    def test_hwc_to_chw_invalid_dims(self):
        
        assert_raises(AttributeError, mu.hwc_to_chw, np.random.rand(3))
        assert_raises(AttributeError, mu.hwc_to_chw, np.random.rand(3, 2))
        assert_raises(AttributeError, mu.hwc_to_chw, np.random.rand(5, 4, 3, 2))
        
    def test_hwc_to_chw(self):
        
        x = np.array([[[ 1,  2,  3],
                       [ 4,  5,  6]
                       ],
                      [[ 7,  8,  9],
                       [10, 11, 12]
                       ],
                      [[13, 14, 15],
                       [16, 17, 18],
                       ],
                      [[19, 20, 21],
                       [22, 23, 24]
                       ]
                      ])
        
        assert_equals(x.shape[0], 4) # h
        assert_equals(x.shape[1], 2) # w
        assert_equals(x.shape[2], 3) # c
        
        y = mu.hwc_to_chw(x)
        
        assert_equals(y.shape[0], 3)
        assert_equals(y.shape[1], 4)
        assert_equals(y.shape[2], 2)
        
        assert_equals(x[3][1][2], y[2][3][1])
        
        for i in range(4):
            for j in range(2):
                for k in range(3):
                    assert_equals(x[i][j][k], y[k][i][j])
                    
class TestExpandDims:
    
    def test_expand_dims_vector(self):
        
        m = np.random.rand(3)
        
        x = mu.expand_dims(np.copy(m), 1)
        assert_equals(x.ndim, 1)
        assert_equals(x.shape, (3,))
        
        x = mu.expand_dims(np.copy(m), 2)
        assert_equals(x.ndim, 2)
        assert_equals(x.shape, (1, 3))
        
        x = mu.expand_dims(np.copy(m), 2)
        assert_equals(x.ndim, 2)
        assert_equals(x.shape, (1, 3))
        
    def test_expand_dims(self):
        
        m = np.random.rand(3, 2)
        for d in xrange(m.ndim+1, m.ndim*3):
            x = mu.expand_dims(np.copy(m), d)
            assert_equals(x.ndim, d)
            assert_equals(x.shape[x.ndim-m.ndim:], m.shape)
            assert_equals(x.shape[:x.ndim-m.ndim], (1,)*(x.ndim-m.ndim))
            
        m = np.random.rand(3, 2, 7)
        for d in xrange(m.ndim+1, m.ndim*3):
            x = mu.expand_dims(np.copy(m), d)
            assert_equals(x.ndim, d)
            assert_equals(x.shape[x.ndim-m.ndim:], m.shape)
            assert_equals(x.shape[:x.ndim-m.ndim], (1,)*(x.ndim-m.ndim))
            
        m = np.random.rand(3, 2, 7, 1, 2)
        for d in xrange(m.ndim+1, m.ndim*3):
            x = mu.expand_dims(np.copy(m), d)
            assert_equals(x.ndim, d)
            assert_equals(x.shape[x.ndim-m.ndim:], m.shape)
            assert_equals(x.shape[:x.ndim-m.ndim], (1,)*(x.ndim-m.ndim))
            
    def test_expand_dims_no_change(self):
        
        m = np.random.rand(3)
        x = mu.expand_dims(np.copy(m), 0)
        assert_equals(x.ndim, m.ndim)
        assert_equals(x.shape, m.shape)
        assert_true(np.any(x == m))
        
        x = mu.expand_dims(np.copy(m), 1)
        assert_equals(x.ndim, m.ndim)
        assert_equals(x.shape, m.shape)
        assert_true(np.any(x == m))
        
        m = np.random.rand(3, 2)
        for d in xrange(m.ndim):
            x = mu.expand_dims(np.copy(m), d)
            assert_equals(x.ndim, m.ndim)
            assert_equals(x.shape, m.shape)
            assert_true(np.any(x == m))
        
        m = np.random.rand(3, 2, 5)
        for d in xrange(m.ndim):
            x = mu.expand_dims(np.copy(m), d)
            assert_equals(x.ndim, m.ndim)
            assert_equals(x.shape, m.shape)
            assert_true(np.any(x == m))
        
        m = np.random.rand(5, 4, 3, 2)
        for d in xrange(m.ndim):
            x = mu.expand_dims(np.copy(m), d)
            assert_equals(x.ndim, m.ndim)
            assert_equals(x.shape, m.shape)
            assert_true(np.any(x == m))