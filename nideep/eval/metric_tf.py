'''
Created on Jul 28, 2017

@author: kashefy
'''
import tensorflow as tf

def resettable_metric(metric, scope, **metric_args):
    '''
    Originally from https://github.com/tensorflow/tensorflow/issues/4814#issuecomment-314801758
    '''
    with tf.variable_scope(scope) as scope:
        metric_op, update_op = metric(**metric_args)
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        reset_op = tf.variables_initializer(vars)
    return metric_op, update_op, reset_op