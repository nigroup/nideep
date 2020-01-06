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
    def _init_weight_op(mean=0.0, stddev=0.1):
        return tf.random_normal_initializer(mean=mean,
                                            stddev=stddev)
    
    @staticmethod
    def _init_bias_op(value):
        return tf.constant_initializer(value)
    
    def vars_new(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope=self.name_scope)
        
    def _init_bias_vars(self, bias_value=0.):
        self.b = {}
        for key, tensor in self.w.iteritems():
            key_b = key.replace('/w', '/b').replace('_w', '_b').replace('-w', '-b')
            if self.reuse:
                self.b[key_b] = self._restore_variable(self.var_scope + '/' + key_b)
            else:    
                self.b[key_b] = tf.get_variable(key_b,
                                                [tensor.get_shape()[-1].value],
                                                initializer=self._init_bias_op(bias_value))
            
    @abstractmethod
    def _init_learning_params_scoped(self):
        pass

    def _init_learning_params(self):
        with tf.variable_scope(self.var_scope, reuse=self.reuse):
            self._init_learning_params_scoped()
        pass
    
    def _restore_variable(self, var_name):
        v = self.get_tensor_by_name(var_name)
        self.vars_restored.append(v)
        return v
    
    def get_tensor_by_name(self, tensor_name, graph=tf.get_default_graph(), index=-1):
        name_with_index = self.get_tensor_names(tensor_name, graph=graph)[index]
        return graph.get_tensor_by_name(name_with_index)
    
    def get_tensor_names(self, tensor_name, graph=tf.get_default_graph()):
#        op_names = [output.name for op in graph.get_operations()
#                    if op.op_def and 'Variable' in op.op_def.name and
#                    tensor_name == op.name for output in op.outputs]
        op_names = [output.name for op in graph.get_operations()
                    if op.op_def and
                    tensor_name == op.name for output in op.outputs]
        return op_names
    
    def __init__(self, params):
        '''
        Constructor
        '''
        self.reuse = params.get('reuse', None)
        super(AbstractNetTF, self).__init__(params)