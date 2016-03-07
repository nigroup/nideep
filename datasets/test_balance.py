'''
Created on Oct 28, 2015

@author: kashefy
'''
from nose.tools import assert_equals, assert_is_instance, \
    assert_list_equal, assert_greater, assert_in, assert_almost_equal,\
    assert_not_in
import numpy as np
from balance import Balancer, CLNAME_OTHER

class TestBalancer:
    
    def setup(self):
        
        # generate fake data
        num_examples = 70
        num_feats = 33
        num_classes = 9 # EXCLUDING other class
        self.x = np.random.randn(num_examples, num_feats)
        self.cl = np.random.randint(0, num_classes, size=(num_examples))
        self.l = np.zeros((num_examples, num_classes))
        self.l[range(num_examples), self.cl] = 1
        
    def test_get_class_count_other_default(self):
        counts = Balancer(np.copy(self.l)).get_class_count(other_clname=CLNAME_OTHER)
        assert_in(CLNAME_OTHER, counts.keys())
    
    def test_get_class_count_other_empty(self):
        
        other_clname = 'other_class_bin'
        counts = Balancer(np.copy(self.l)).get_class_count(other_clname=other_clname)
        assert_is_instance(counts, dict, "Unexpected return instance type.")
        assert_equals(len(counts.keys()), self.l.shape[-1]+1,
                      "Expecting a key for each class + 1 for 'other'.")
        
        assert_in(other_clname, counts.keys())
        
        for key in counts.keys():
            if key == other_clname:
                assert_equals(counts[key], 0,
                              "Unexpected count for 'other' class")
            else:
                assert_equals(counts[key], np.sum(self.l[:, int(key)]),
                              "Unexpected count for class '%s'" % (key,))
                
    def test_get_class_count_other_non_empty(self):
        
        other_clname = 'foo'
        n, num_classes = self.l.shape
        # append label vector for 'other' class
        labls = np.vstack((self.l,
                           np.zeros((n*2, num_classes), dtype=self.l.dtype)))
        counts = Balancer(labls).get_class_count(other_clname=other_clname)
        assert_is_instance(counts, dict, "Unexpected return instance type.")
        assert_equals(len(counts.keys()), self.l.shape[-1]+1,
                      "Expecting a key for each class + 1 for 'other'.")
        
        assert_in(other_clname, counts.keys())
        
        for key in counts.keys():
            if key == other_clname:
                assert_equals(counts[key], n*2,
                              "Unexpected count for '%s' class" % (other_clname,))
            else:
                assert_equals(counts[key], np.sum(self.l[:, int(key)]),
                              "Unexpected count for class '%s'" % (key,))
        
    def test_get_class_count_no_other(self):
        
        counts = Balancer(np.copy(self.l)).get_class_count(other_clname=None)
        assert_is_instance(counts, dict, "Unexpected return instance type.")
        assert_list_equal(counts.keys(), range(self.l.shape[-1]),
                          "Expecting a key for each class.")
        for key in counts.keys():
            assert_equals(counts[key], np.sum(self.l[:, int(key)]),
                          "Unexpected count for class '%s'" % (key,))
        
        assert_greater(np.sum(counts.values()), 0)
        
class TestBalancerBalanceIdxs:
    
    def setup(self):
        
        # generate fake data
        num_examples = 100
        num_classes = 2 # EXCLUDING other class
        self.l = np.zeros((num_examples, num_classes))
        self.l[0:10, 0] = 1
        self.l[10:60, 1] = 1
    
    def test_get_idxs_to_balance_class_count_other_not_highest(self):
        
        bal = Balancer(np.copy(self.l))
        counts = bal.get_class_count(other_clname=CLNAME_OTHER)
        assert_in(CLNAME_OTHER, counts.keys())
        
        assert_equals(counts[0], 10)
        assert_equals(counts[1], 50)
        assert_equals(counts[CLNAME_OTHER], 40)
        tolerance_order = 1
        idxs = bal.get_idxs_to_balance_class_count(counts.values())
        assert_almost_equal(np.count_nonzero(np.logical_and(idxs >= 0,
                                                            idxs < 10)
                                             ),
                            10+(50-10), tolerance_order)
        assert_equals(np.count_nonzero(np.logical_and(idxs >= 10,
                                                      idxs < 60)
                                       ),
                      50, 1)
        assert_almost_equal(np.count_nonzero(idxs >= 60),
                            40+(50-40), tolerance_order)
        
    def test_get_idxs_to_balance_class_count_other_highest(self):
        
        self.l[10:60, 1] = 0
        self.l[10:30, 1] = 1
        bal = Balancer(np.copy(self.l))
        counts = bal.get_class_count(other_clname=CLNAME_OTHER)
        assert_in(CLNAME_OTHER, counts.keys())
        
        assert_equals(counts[0], 10)
        assert_equals(counts[1], 20)
        assert_equals(counts[CLNAME_OTHER], 70)
        assert_equals(counts[CLNAME_OTHER], np.max(counts.values()),
                      "this test requires class count for %s to be highest!")
        tolerance_order = 1
        idxs = bal.get_idxs_to_balance_class_count(counts.values())
        assert_almost_equal(np.count_nonzero(np.logical_and(idxs >= 0,
                                                            idxs < 10)
                                             ),
                            10+(70-10), tolerance_order)
        assert_almost_equal(np.count_nonzero(np.logical_and(idxs >= 10,
                                                            idxs < 30)
                                             ),
                            20+(70-20), tolerance_order)
        assert_equals(np.count_nonzero(idxs >= 30),
                      70, tolerance_order)
        
    def test_get_idxs_to_balance_class_count_no_other(self):
        
        new_col = np.zeros( (len(self.l), 1) )
        labls = np.hstack( (self.l, new_col) )
        labls[60:, -1] = 1
        bal = Balancer(labls)
        counts = bal.get_class_count(other_clname=None)
        assert_not_in(CLNAME_OTHER, counts.keys())
        
        assert_equals(counts[0], 10)
        assert_equals(counts[1], 50)
        assert_equals(counts[2], 40)
        tolerance_order = 1
        idxs = bal.get_idxs_to_balance_class_count(counts.values())
        assert_almost_equal(np.count_nonzero(np.logical_and(idxs >= 0,
                                                            idxs < 10)
                                             ),
                            10+(50-10), tolerance_order)
        assert_equals(np.count_nonzero(np.logical_and(idxs >= 10,
                                                      idxs < 60)
                                       ),
                      50, 1)
        assert_almost_equal(np.count_nonzero(idxs >= 60),
                            40+(50-40), tolerance_order)
