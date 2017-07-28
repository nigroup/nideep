'''
Created on Jul 28, 2017

@author: kashefy
'''
from nose.tools import assert_equals
import tensorflow as tf
import metric_tf as m

class TestResettableMetric:

    def test_resettable_metric(self):
       
        x = tf.placeholder(tf.int32, [None, 4])
        y = tf.placeholder(tf.int32, [None, 4])
        m_op, up_op, reset_op = m.resettable_metric(tf.metrics.accuracy,
                                                     'foo',
                                                     labels=x, predictions=y)
    
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        x1 = [[1, 0, 0, 0]]
        y1 = [[1, 0, 0, 1]]
    
        _, out_up = sess.run([m_op, up_op],
                             feed_dict={x: x1,
                                        y: y1})
        assert_equals(3/4., out_up)
        assert_equals(3/4., sess.run(m_op))
        
        x2 = [[1, 0, 0, 0]]
        y2 = [[0, 1, 1, 1]]
        _, out_up = sess.run([m_op, up_op],
                             feed_dict={x: x2,
                                        y: y2})
        assert_equals(3/8., out_up)
        assert_equals(3/8., sess.run(m_op))
        
        _, out_up = sess.run([m_op, up_op],
                             feed_dict={x: x2,
                                        y: y2})
        assert_equals(3/12., out_up)
        assert_equals(3/12., sess.run(m_op))
        
        sess.run([reset_op])
    
        _, _,out_up = sess.run([reset_op, m_op, up_op],
                             feed_dict={x: x2,
                                        y: y2})
        assert_equals(0., out_up)
        assert_equals(0., sess.run(m_op))
        
        _, out_up = sess.run([m_op, up_op],
                             feed_dict={x: x1,
                                        y: y1})
        assert_equals(3/8., out_up)
        assert_equals(3/8., sess.run(m_op))