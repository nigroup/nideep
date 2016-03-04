'''
Created on Oct 28, 2015

@author: kashefy
'''
from nose.tools import assert_equals, assert_is_instance, assert_is_not_none, \
    assert_list_equal
import numpy as np
from balance import Balancer, get_class_count_hdf5

class TestBalancer:
    
    def test_balancer(self):
        fpath = '/home/kashefy/data/x/labels_test.h5'
        print get_class_count_hdf5(fpath)