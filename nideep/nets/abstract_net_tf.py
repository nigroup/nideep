'''
Created on Jul 18, 2017

@author: kashefy
'''
import tensorflow as tf
from nideep.nets.abstract_net import AbstractNet
from abc import ABCMeta, abstractmethod

class AbstractNetTF(AbstractNet):
    '''
    classdocs
    '''
    __metaclass__ = ABCMeta
    
    @staticmethod
    def _init_weight_op():
        return tf.random_normal_initializer()
    
    @staticmethod
    def _init_bias_op(value):
        return tf.constant_initializer(value)
    
    def vars_new(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope=self.name_scope)
        
    @abstractmethod
    def _init_learning_params_scoped(self):
        pass

    def _init_learning_params(self):
        with tf.variable_scope(self.var_scope):
            self._init_learning_params_scoped()
        pass
    
    def __init__(self, params):
        '''
        Constructor
        '''
        super(AbstractNetTF, self).__init__(params)