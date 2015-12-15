'''
Created on Dec 3, 2015

@author: kashefy
'''
import numpy as np
import h5py
import caffe
import read_lmdb
import to_lmdb

def infer_to_h5_fixed_dims(net, keys, n, dst_fpath, preserve_batch=False):
    """
    Run network inference for n batches and save results to file
    """
    dc = {k:[] for k in keys}
    for _ in range(n):
        d = forward(net, keys)
        for k in keys:
            if preserve_batch:
                dc[k].append(np.copy(d[k]))
            else:
                dc[k].extend(np.copy(d[k]))
            
    with h5py.File(dst_fpath, "w") as f:
        for k in keys:
            f[k] = dc[k]
            
    return [len(dc[k]) for k in keys]

def infer_to_lmdb(net, keys, n, dst_fpath, preserve_batch=False):
    """
    Run network inference for n batches and save results to file
    """
    dc = {k:[] for k in keys}
    for _ in range(n):
        d = forward(net, keys)
        for k in keys:
            if preserve_batch:
                dc[k].append(np.copy(d[k].astype(float)))
            else:
                dc[k].extend(np.copy(d[k].astype(float)))
          
    for k in keys:
        to_lmdb.arrays_to_lmdb(dc[k], dst_fpath)
            
    return [len(dc[k]) for k in keys]

def forward(net, keys):
    
    net.forward()
    return {k : net.blobs[k].data for k in keys}

if __name__ == '__main__':
    
    from os.path import expanduser
        
    fpath_net = expanduser('~/models/dark/mnist/t0/lenet_train_test_inf.prototxt')
    
    fpath_weights = expanduser('~/models/dark/mnist/t0/lenet_iter_10000.caffemodel')
    net = caffe.Net(fpath_net, fpath_weights, caffe.TRAIN)
    
    fpath = expanduser('~/models/dark/mnist/t0/mnist_dark0_sm_train')
    num_entries = read_lmdb.num_entries("/home/kashefy/data/mnist/mnist_train_lmdb")

    
    #infer_to_h5_fixed_dims(net, ['prob'], num_entries, "%s.h5" % fpath)
    infer_to_lmdb(net, ['prob'], num_entries, "%s_lmdb" % fpath)
    
    net = caffe.Net(fpath_net, fpath_weights, caffe.TEST)
    fpath = expanduser('~/models/dark/mnist/t0/mnist_dark0_sm_test')
    num_entries = read_lmdb.num_entries("/home/kashefy/data/mnist/mnist_test_lmdb")
    #infer_to_h5_fixed_dims(net, ['prob'], num_entries, "%s.h5" % fpath)
    infer_to_lmdb(net, ['prob'], num_entries, "%s_lmdb" % fpath)
    
    print 'done'
    
    #l = read_lmdb.read_labels('/home/kashefy/data/mnist/mnist_train_lmdb')
    #p = [x for x, _ in read_lmdb.read_values("~/models/dark/mnist/t0/mnist_dark0_sm_train_lmdb")]
    
    
#     with h5py.File(fpath, "w") as f:
#     
#         f['a'] = 0
#         
#         
#         f['b'] = [1, 2]
#         f['c'] = np.arange(3)
#         f['d'] = [np.array([[1,2],[4,5]], dtype=float), np.array([[1,2],[4, 5]], dtype=float)+10]
            
    #infer_to_h5(net, 1, ['accuracy'], fpath)
    
    pass