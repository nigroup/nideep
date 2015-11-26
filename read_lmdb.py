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
        for _, value in txn.cursor():
            
            dat = caffe.proto.caffe_pb2.Datum()
            dat.ParseFromString(value)
            labels.append(dat.label)
            
    return labels

def unpack_raw_datum(raw_datum, dtype=None):
    """
    Wraps de-serialization instructions
    returns datum object and array values
    """
    dat = caffe.proto.caffe_pb2.Datum()
    dat.ParseFromString(raw_datum)
    
    channels = dat.channels  # assignment makes mocking easier
    height = dat.height  # assignment makes mocking easier
    width = dat.width  # assignment makes mocking easier
    
    d = dat.data     # assignment makes mocking easier
    df = dat.float_data # assignment makes mocking easier
    
    if dtype is not None and (dtype in [float, np.float]):
        x = np.array(df).astype(float)
    elif d == '' and df is not None:
        x = np.array(df).astype(float)
    else:
        x = np.fromstring(d, dtype=np.uint8)

    x = x.reshape(channels, height, width) # restore shape
    
    return dat, x

def read_values(path_lmdb, dtype=None):
    """
    Read label member from lmdb
    adapted from Gustav Larsson http://deepdish.io/2015/04/28/creating-lmdb-in-python/
    """
    v = []
    with lmdb.open(path_lmdb, readonly=True).begin() as txn:
        for _, value in txn.cursor():
            
            dat, x = unpack_raw_datum(value, dtype)
            y = dat.label # assume scalar
            v.append((x, y))
            
    return v

def read_values_at(path_lmdb, key, dtype=None):
    """
    Read key from lmdb
    adapted from Gustav Larsson http://deepdish.io/2015/04/28/creating-lmdb-in-python/
    """
    with lmdb.open(path_lmdb, readonly=True).begin() as txn:
            
        if not isinstance(key, basestring):
            key = b'{:0>10d}'.format(key)
        dat, x = unpack_raw_datum(txn.get(key), dtype)
        return x, dat.label # scalar label
    
# CREDIT: Gustav Larsson http://deepdish.io/2015/04/28/creating-lmdb-in-python/
if __name__ == '__main__':
    
    import os
    path_lmdb = os.path.expanduser('~/src/caffe/examples/mnist/mnist_train_lmdb')
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
    pass