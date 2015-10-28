'''
Created on Oct 28, 2015

@author: kashefy
'''
from nose.tools import assert_equals, assert_is_instance, assert_is_not_none, \
    assert_list_equal
import dataset_utils as du

class TestTrainValSplit:


    def test_split_empty(self):
        
        train_idx, val_idx = du.get_train_val_split([], [])
        
        assert_is_not_none(train_idx)
        assert_is_instance(train_idx, list)
        assert_equals(len(train_idx), 0)
                
        assert_is_not_none(val_idx)
        assert_is_instance(val_idx, list)
        assert_equals(len(val_idx), 0)
        
    def test_split_ret_type(self):
        
        train_idx, val_idx = du.get_train_val_split(['a', 'b', 'c'], ['b'])
        
        assert_is_not_none(train_idx)
        assert_is_instance(train_idx, list)
                
        assert_is_not_none(val_idx)
        assert_is_instance(val_idx, list)
        
    def test_split(self):
        
        train_idx, val_idx = du.get_train_val_split(['a', 'b', 'c'], ['b'])
        
        assert_list_equal(train_idx, [0, 2])
        assert_list_equal(val_idx, [1])


