'''
Created on Jan 06, 2016

@author: kashefy
'''
from nose.tools import assert_greater, assert_equal, assert_is_instance, assert_true
import lmdb_utils as lu

class TestLMDBConsts:
            
    def test_map_sz(self):
        
        assert_greater(lu.MAP_SZ, 0)
        assert_is_instance(lu.MAP_SZ, int)
        
    def test_num_idx_digits(self):
        
        assert_greater(lu.NUM_IDX_DIGITS, 0)
        assert_is_instance(lu.NUM_IDX_DIGITS, int)
        
class TestIdxFormat:
        
    def test_idx_format(self):
        
        assert_greater(len(lu.IDX_FMT), 0)
        assert_is_instance(lu.IDX_FMT, str)
        assert_true(lu.IDX_FMT.startswith('{'))
        assert_true(lu.IDX_FMT.endswith('}'))
        
    def test_idx_format_zero(self):
        
        assert_equal(lu.IDX_FMT.format(0), ''.join(['0']*lu.NUM_IDX_DIGITS))
        
    def test_idx_format_nonzero(self):
        
        assert_greater(lu.NUM_IDX_DIGITS, 1)
        
        s = ''.join(['0']*lu.NUM_IDX_DIGITS)
        
        for i in xrange(10):
            assert_equal(lu.IDX_FMT.format(i), s[:-1] + '%d' % i)
        
        for i in xrange(10, 100):
            assert_equal(lu.IDX_FMT.format(i), s[:-2] + '%d' % i)
        