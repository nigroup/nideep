'''
Created on Sep 16, 2015

@author: kashefy
'''

import os
import numpy as np
import lmdb

CAFFE_ROOT = '/home/kashefy/src/caffe/'
if CAFFE_ROOT is not None:
    import sys
    sys.path.insert(0,  os.path.join(CAFFE_ROOT, 'python'))
import caffe
    
# CREDIT: Gustav Larsson http://deepdish.io/2015/04/28/creating-lmdb-in-python/
if __name__ == '__main__':
    
    path_lmdb = '/home/kashefy/src/caffe/examples/mnist/mnist_train_lmdb'
    env = lmdb.open(path_lmdb, readonly=True)
    
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            #print(key, value)
            
            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(value)
             
            flat_x = np.fromstring(datum.data, dtype=np.uint8)
            x = flat_x.reshape(datum.channels, datum.height, datum.width)
            y = datum.label
            
            print y
    
    print y

    pass