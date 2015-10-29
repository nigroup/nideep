'''
Created on Oct 28, 2015

@author: kashefy
'''
from nose.tools import assert_equals, assert_raises
import numpy as np
import mat_utils as mu

class TestTranspose:

    def test_cwh_to_chw_invalid_dims(self):
        
        assert_raises(AttributeError, mu.cwh_to_chw, np.random.rand(3))
        assert_raises(AttributeError, mu.cwh_to_chw, np.random.rand(3, 2))
        assert_raises(AttributeError, mu.cwh_to_chw, np.random.rand(5, 4, 3, 2))
        
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
        
        assert_equals(x.shape[0], 4)
        assert_equals(x.shape[1], 2)
        assert_equals(x.shape[2], 3)
        
        y = mu.hwc_to_chw(x)
        
        assert_equals(y.shape[0], 3)
        assert_equals(y.shape[1], 4)
        assert_equals(y.shape[2], 2)
        
        assert_equals(x[3][1][2], y[2][3][1])
        
        for i in range(4):
            for j in range(2):
                for k in range(3):
                    assert_equals(x[i][j][k], y[k][i][j])