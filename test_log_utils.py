'''
Created on Oct 1, 2015

@author: kashefy
'''

from nose.tools import assert_equal, assert_is_instance
import log_utils as lu

def test_pid_from_logname():
    
    s = 'caffe.host.user.log.INFO.20151001-132750.26943'
    res = lu.pid_from_logname(s)
    assert_is_instance(res, int)
    assert_equal(res, 26943)
    