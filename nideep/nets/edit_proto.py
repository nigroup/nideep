'''
Created on Sep 21, 2015

@author: kashefy
'''
import caffe
from caffe import layers as L
from caffe import params as P

from nideep.proto import proto_utils as pu

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

def edit_net_params(path_src, target, path_dst):
    
    netfile_parser = pu.Parser()
    x = netfile_parser.from_net_params_file(path_src)
    
    print len(x.layer)
    count = 0
    for l in x.layer:
        if target.name == l.name:
            print "Layer %s found" % l.name
            print [name for name in dir(l) if not name.startswith('__')]
            l.data_param.batch_size = target.new_value
            count += 1
            print l
    
    return count
    
    return
    n = caffe.Net(path_src, caffe.TEST)
    
    batch_size = 10
    lmdb = 'lmdb'
    
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.ip1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    
    t = n.tops
    
    print t.keys()
    
    #print n.to_proto()

def run_edit_net_params():
    
    path_src = 'lenet_train_test.prototxt'
    path_dst = 'lenet_train_testX.prototxt'
    
    name = 'mnist'
    key = {'data_param' : 'batch_size'}
    new_value = 128
    
    target = pu.Target(name, key, new_value)
    
    count = edit_net_params(path_src, target, path_dst)
    
    print "Target edited %d times" % count

if __name__ == '__main__':
    
    run_edit_net_params()
    pass