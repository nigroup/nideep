'''
Created on Jul 14, 2017

@author: kashefy
'''
import tensorflow as tf
from nideep.nets.abstract_net_tf import AbstractNetTF

class MLP(AbstractNetTF):
    '''
    classdocs
    '''
    def _init_learning_params_scoped(self):
        self.w = {}
        for idx, dim in enumerate(self.n_nodes):
            input_dim = self.n_nodes[idx-1]
            if idx == 0:
                input_dim = self.n_input    
            fc_name_w = 'fc-%d/w' % idx
            self.w[fc_name_w] = tf.get_variable(fc_name_w,
                                                [input_dim, dim],
                                                initializer=self._init_weight_op()
                                                )
            if idx == len(self.n_nodes)-1:
                fc_name_w = 'fc-%d-aux/w' % idx
                self.w[fc_name_w] = tf.get_variable(fc_name_w,
                                                    [input_dim, 9],
                                                    initializer=self._init_weight_op()
                                                    )
        self._init_bias_vars(bias_value=0.1)
                
    def _fc(self, x):
        in_op = x
        for idx in xrange(len(self.n_nodes)):
            fc_name_w = 'fc-%d/w' % idx
            fc_name_b = 'fc-%d/b' % idx
            fc_op = tf.add(tf.matmul(in_op, self.w[fc_name_w]),
                            self.b[fc_name_b],
                            name='fc-%d' % idx)
            if idx < len(self.n_nodes)-1:
                a = tf.sigmoid(fc_op, name='a-%d' % idx)
                in_op = a
            else:
                self._y_logits = fc_op
                self.p = tf.nn.softmax(fc_op, name='a-%d' % idx)
        
        fc_name_w = 'fc-%d-aux/w' % idx
        fc_name_b = 'fc-%d-aux/b' % idx
        if len(self.n_nodes) == 1:
            in_op = x
        else:
            in_op = a
        fc_op = tf.add(tf.matmul(in_op, self.w[fc_name_w]),
                        self.b[fc_name_b],
                        name='fc-%d' % idx)
        self._y_logits_aux = fc_op
        self.p_aux = tf.nn.softmax(fc_op, name='a-%d-aux' % idx)
        return self.p, self._y_logits
            
    def _init_ops(self):
        with tf.name_scope(self.name_scope + 'fc'):
            self.p, self.logits = self._fc(self.x)
                
    def cost(self, y_true, name=None):
        with tf.name_scope(self.name_scope):
            c = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(\
                        labels=y_true, logits=self._y_logits),
                        name=name)
            self._cost_ops.append(c)
            return c
        
    def cost_o(self, y_true, name=None):
        with tf.name_scope(self.name_scope):
            c = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(\
                        labels=y_true, logits=self._y_logits_aux),
                        name=name)
            self._cost_ops.append(c)
            return c
        
    @property
    def cost_ops(self):
        return self._cost_ops
        
    def __init__(self, params):
        '''
        Constructor
        '''
        # Network Parameters
        self.n_input = params['n_input']  # 1st layer num features
        self.n_nodes = params['n_nodes']  # 1st layer num features
        super(MLP, self).__init__(params)

        self._cost_ops = []
        self._y_logits = None