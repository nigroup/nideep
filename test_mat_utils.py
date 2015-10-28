'''
Created on Oct 28, 2015

@author: kashefy
'''
from nose.tools import assert_equals
import numpy as np

class TestTranspose:

    def test_whc_to_chw(self):
        
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