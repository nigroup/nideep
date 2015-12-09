'''
Created on Dec 3, 2015

@author: kashefy
'''
import numpy as np
import h5py
import caffe

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

def forward(net, keys):
    
    net.forward()
    return {k : net.blobs[k].data for k in keys}

if __name__ == '__main__':
    
    import os
    fpath_net = os.path.expanduser('~/models/dark/mnist/t0/lenet_train_test.prototxt')
    fpath_weights = os.path.expanduser('~/models/dark/mnist/t0/lenet_iter_10000.caffemodel')
    net = caffe.Net(fpath_net, fpath_weights, caffe.TEST)
    fpath = os.path.expanduser('~/models/dark/mnist/t0/x.h5')
    infer_to_h5_fixed_dims(net, ['data', 'label'], 4, fpath)
    
    
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