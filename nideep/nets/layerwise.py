'''
Created on Jan 21, 2016

@author: kashefy
'''
from google.protobuf import text_format
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto.caffe_pb2 import NetParameter, LayerParameter
import nideep.proto.proto_utils as pu

def stack(net, level):
    
    n = pu.copy_net_params(net)
    
    enc_prev = None
    dec_prev = None
    enc = None
    dec = None
    for l in n.layer:
        if l.name.lower().endswith('encode%03dneuron' % (level-1)):
            enc = pu.copy_msg(l, LayerParameter)
            
            for b in list(enc.bottom):
                l.bottom.remove(b)
            for t in list(l.top):
                l.bottom.append(unicode(t)) # preserve order of layer bottoms, label as bottom has to come last
            
        elif l.name.lower().endswith('decode%03dneuron' % (level-1)):
            dec_prev = l
            dec = pu.copy_msg(l, LayerParameter)
            
    enc.name = 'encode%03dneuron' % level
    dec.name = 'encode%03dneuron' % level
    
    
    return n

def base_ae(src_train, src_test):
    
    n = caffe.NetSpec()
    
    n.data = \
        L.Data(batch_size=100,
               backend=P.Data.LMDB,
               source=src_train,
               transform_param=dict(scale=0.0039215684),
               ntop=1,
               include=[dict(phase=caffe.TRAIN)]
               )
    
    n.data_test = \
        L.Data(batch_size=100,backend=P.Data.LMDB,
               source=src_test,
               transform_param=dict(scale=0.0039215684),
               ntop=1,
               include=[dict(phase=caffe.TEST)]
               )
                      
    n.flatdata = L.Flatten(n.data)
                              
    n.encode001 = \
        L.InnerProduct(n.data,
                       num_output=64,
                       param=[dict(lr_mult=1, decay_mult=1),
                              dict(lr_mult=1, decay_mult=0)],
                       weight_filler=dict(type='gaussian', std=1, sparse=15),
                       bias_filler=dict(type='constant', value=0)
                       )
     
    n.encode001neuron = L.Sigmoid(n.encode001, in_place=True)
    
    n.decode001 = L.InnerProduct(n.encode001neuron,
                                 num_output=3072,
                                 param=[dict(lr_mult=1, decay_mult=1),
                                 dict(lr_mult=1, decay_mult=0)],
                                 weight_filler=dict(type='gaussian', std=1, sparse=15),
                                 bias_filler=dict(type='constant', value=0)
                                 )
    n.loss_x_entropy = \
        L.SigmoidCrossEntropyLoss(n.decode001, n.flatdata,
                                  loss_weight=[1])
        
    n.decode001neuron = L.Sigmoid(n.decode001, in_place=False)
    
    n.loss_l2 = \
        L.EuclideanLoss(n.decode001neuron, n.flatdata,
                        loss_weight=[0])

    n_proto = n.to_proto()
    
    # fix data layer for test phase
    for l in n_proto.layer:
        if l.type.lower() == 'data' and \
           [x.phase for x in l.include] == [caffe.TEST]:
            for t in list(l.top):
                l.top.remove(t)
                t = t.split('_test')[0]
                l.top.append(unicode(t))
            l.name = l.name.split('_test')[0]
    
    return n_proto

if __name__ == '__main__':
    
    src_train = "/home/kashefy/data/cifar10/cifar10_train_lmdb"
    src_test = "/home/kashefy/data/cifar10/cifar10_test_lmdb"
    n = base_ae(src_train, src_test)
    s = text_format.MessageToString(n)
    #print s
    
    stack(n, 2)
    
    pass