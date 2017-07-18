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
        fc_name_w = 'fc-%d/w' % self.depth
        self.w = {
            fc_name_w: tf.get_variable(fc_name_w,
                                       [self.n_hidden[-1], self.n_outputs],
                                       initializer=tf.random_normal_initializer(),
                                       )
        }
        self.b = {}
        for key, value in self.w.iteritems():
            key_b = key.replace('/w', '/b').replace('_w', '_b').replace('-w', '-b')
            self.b[key_b] = tf.get_variable(\
                                            key_b,
                                            [int(value.get_shape()[-1])],
                                            initializer=tf.constant_initializer(0.)
                                            )
                
    def _fc(self, x):
        fc_name_w = 'fc-%d/w' % self.depth
        fc_name_b = 'fc-%d/b' % self.depth
        fc_op = tf.add(tf.matmul(x, self.w[fc_name_w]),
                        self.b[fc_name_b],
                        name='fc-%d' % self.depth)
        self._y_logits = fc_op
        self.y_pred = tf.nn.softmax(fc_op, name='a-%d' % self.depth)
        return self.y_pred, self._y_logits
            
    def build(self):
        with tf.name_scope(self.name_scope + 'fc'):
            pred, logits = self._fc(self.x)
        return pred, logits
                
    def cost(self, y_true, name=None):
        with tf.name_scope(self.name_scope):
            self._cost_op = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(\
                        labels=y_true, logits=self._y_logits),
                        name=name)
            return self._cost_op
        
    def __init__(self, params):
        '''
        Constructor
        '''
        # Network Parameters
        self.n_hidden = params['n_hidden']  # 1st layer num features
        self.n_outputs = params['n_outputs']  # 1st layer num features
        super(MLP, self).__init__(params)

        self._cost_op = None
        self._y_logits = None