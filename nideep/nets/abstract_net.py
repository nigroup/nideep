'''
Created on Jul 14, 2017

@author: kashefy
'''
from __future__ import division, print_function, absolute_import
from abc import ABCMeta, abstractmethod

class AbstractNet(object):
    '''
    classdocs
    '''
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def _init_learning_params(self):
        pass
    
    @abstractmethod
    def _init_ops(self):
        pass # return prediction_op, logit_op
    
    def build(self):
        self._init_learning_params()
        self._init_ops()
        return self.p, self.logits
    
    def _config_scopes(self):
        self.var_scope = self.prefix
        self.name_scope = self.var_scope + '/'
    
    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @x.deleter
    def x(self):
        del self._x
        
    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        self._p = value

    @p.deleter
    def p(self):
        del self._p
        
    @property
    def logits(self):
        return self._logits

    @logits.setter
    def logits(self, value):
        self._logits = value

    @logits.deleter
    def logits(self):
        del self._logits
        
    @property
    def vars_restored(self):
        return self._vars_restored
    
    @vars_restored.setter
    def vars_restored(self, value):
        self._vars_restored = value

    @vars_restored.deleter
    def vars_restored(self):
        del self._vars_restored

    def __init__(self, params):
        '''
        Constructor
        '''
        self._x = None
        self._p = None
        self.prefix = params.get('prefix', '')
        self._config_scopes()
        self._vars_restored = []
        self._cost_op = None
        