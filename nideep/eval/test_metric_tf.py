'''
Created on Jul 28, 2017

@author: kashefy
'''
from nose.tools import assert_equals
import tensorflow as tf
import metric_tf as m

class TestResettableMetric:

    def test_resettable_metric(self):
        