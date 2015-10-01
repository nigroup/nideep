'''
Created on Oct 1, 2015

@author: kashefy
'''

from nose.tools import assert_equal, assert_is_instance
import log_utils as lu

def test_pid_from_str():
    
    n = 26943
    res = lu.pid_from_str('%s' % n)
    assert_is_instance(res, int)
    assert_equal(res, n)
    
def test_pid_from_str_neg():
    
    n = -26943
    res = lu.pid_from_str('%s' % n)
    assert_is_instance(res, int)
    assert_equal(res, n)
    
def test_pid_from_str_invalid_alpha():
    
    n = 26943
    x = ['a%s' % n, '%sa' % n, '269a43']
    
    for s in x:
    
        res = lu.pid_from_str(s)
        assert_is_instance(res, int)
        assert_equal(res, -1)
    
def test_pid_from_logname():
    
    s = 'caffe.host.user.log.INFO.20151001-132750.26943'
    res = lu.pid_from_logname(s)
    assert_is_instance(res, int)
    assert_equal(res, 26943)
    
    n = 77775
    s = 'log.%s' % n
    res = lu.pid_from_logname(s)
    assert_is_instance(res, int)
    assert_equal(res, n)
    
def test_pid_from_logname_invalid():
    
    s = 'caffe.host.user.log.INFO.20151001-132750.26943.txt'
    res = lu.pid_from_logname(s)
    assert_is_instance(res, int)
    assert_equal(res, -1)
    