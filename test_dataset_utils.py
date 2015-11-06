'''
Created on Oct 28, 2015

@author: kashefy
'''
from nose.tools import assert_equals, assert_is_instance, assert_is_not_none, \
    assert_list_equal
import dataset_utils as du

class TestTrainValSplitFromNames:


    def test_split_empty(self):
        
        train_idx, val_idx = du.get_train_val_split_from_names([], [])
        
        assert_is_not_none(train_idx)
        assert_is_instance(train_idx, list)
        assert_equals(len(train_idx), 0)
                
        assert_is_not_none(val_idx)
        assert_is_instance(val_idx, list)
        assert_equals(len(val_idx), 0)
        
    def test_split_ret_type(self):
        
        train_idx, val_idx = du.get_train_val_split_from_names(['a', 'b', 'c'], ['b'])
        
        assert_is_not_none(train_idx)
        assert_is_instance(train_idx, list)
                
        assert_is_not_none(val_idx)
        assert_is_instance(val_idx, list)
        
    def test_split(self):
        
        train_idx, val_idx = du.get_train_val_split_from_names(['a', 'b', 'c'], ['b'])
        
        assert_list_equal(train_idx, [0, 2])
        assert_list_equal(val_idx, [1])

class TestTrainValSplitFromIdx:


    def test_split_empty(self):
        
        train_idx, val_idx = du.get_train_val_split_from_idx([], [])
        
        assert_is_not_none(train_idx)
        assert_is_instance(train_idx, list)
        assert_equals(len(train_idx), 0)
                
        assert_is_not_none(val_idx)
        assert_is_instance(val_idx, list)
        assert_equals(len(val_idx), 0)
        
    def test_split_ret_type(self):
        
        train_idx, val_idx = du.get_train_val_split_from_idx([0, 1, 2], [1])
        
        assert_is_not_none(train_idx)
        assert_is_instance(train_idx, list)
                
        assert_is_not_none(val_idx)
        assert_is_instance(val_idx, list)
        
    def test_split(self):
        
        train_idx, val_idx = du.get_train_val_split_from_idx([0, 1, 2], [1])
        
        assert_list_equal(train_idx, [0, 2])
        assert_list_equal(val_idx, [1])
        
    def test_split_outside_range(self):
        
        train_idx, val_idx = du.get_train_val_split_from_idx([0, 1, 2], [4])
        
        assert_list_equal(train_idx, [0, 1, 2])
        assert_list_equal(val_idx, [4])

    def test_split_not_iterable(self):
        
        train_idx, val_idx = du.get_train_val_split_from_idx(3, [1])
        
        assert_list_equal(train_idx, [0, 2])
        assert_list_equal(val_idx, [1])
        
    def test_split_not_iterable_outside_range(self):
        
        train_idx, val_idx = du.get_train_val_split_from_idx(3, [40])
        
        assert_list_equal(train_idx, [0, 1, 2])
        assert_list_equal(val_idx, [40])
        
    def test_split_empty_val(self):
        
        train_idx, val_idx = du.get_train_val_split_from_idx([0, 1, 2], [])
        
        assert_list_equal(train_idx, [0, 1, 2])
        assert_list_equal(val_idx, [])
