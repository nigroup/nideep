'''
Created on Sep 16, 2015

@author: kashefy
'''
import numpy as np
import lmdb
import caffe

def read_labels(path_lmdb):
    """
    Read label member from lmdb
    adapted from Gustav Larsson http://deepdish.io/2015/04/28/creating-lmdb-in-python/
    """
    
    labels = []
    with lmdb.open(path_lmdb, readonly=True).begin() as txn:
        cursor = txn.cursor()
        for _, value in cursor:
            
            dat = caffe.proto.caffe_pb2.Datum()
            dat.ParseFromString(value)
            labels.append(dat.label)
            
    return labels
    
# CREDIT: Gustav Larsson http://deepdish.io/2015/04/28/creating-lmdb-in-python/
if __name__ == '__main__':
    
    path_lmdb = '/home/kashefy/src/caffe/examples/mnist/mnist_train_lmdb'
    env = lmdb.open(path_lmdb, readonly=True)
    
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            #print(key, value)
            
            dat = caffe.proto.caffe_pb2.Datum()
            dat.ParseFromString(value)
             
            flat_x = np.fromstring(dat.data, dtype=np.uint8)
            x = flat_x.reshape(dat.channels, dat.height, dat.width)
            y = dat.label
            
            print y
    
    print y

    pass