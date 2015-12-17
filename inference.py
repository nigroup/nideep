'''
Created on Dec 3, 2015

@author: kashefy
'''
import numpy as np
import h5py
import caffe
import read_lmdb
import to_lmdb
from proto_utils import Parser

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

def infer_to_lmdb(net, keys, n, dst_prefix, preserve_batch=False):
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
        to_lmdb.arrays_to_lmdb(dc[k], dst_prefix % k)
            
    return [len(dc[k]) for k in keys]

def forward(net, keys):
    
    net.forward()
    return {k : net.blobs[k].data for k in keys}

def est_min_num_fwd_passes(fpath_net, mode_str):
    """
    if multiple source for same mode, base num_passes on last
    fpath_net -- path to network definition
    mode_str -- train or test?
    
    return
    minimum no. of forward passes to cover training set 
    """
    np = Parser().from_net_params_file(fpath_net)
    
    num_passes = 0
    
    for l in np.layer:
        if 'data' in l.type.lower() and mode_str.lower() in l.data_param.source.lower():
            print l.data_param.batch_size, l.phase
            num_entries = read_lmdb.num_entries(l.data_param.source)
            num_passes = int(num_entries / l.data_param.batch_size)
            if num_entries % l.data_param.batch_size != 0:
                print("WARNING: db size not a multiple of batch size. Adding another fwd. pass.")
                num_passes += 1
            print("%d fwd. passes with batch size %d" % (num_passes, l.data_param.batch_size))
            
    return num_passes
            
def response_to_lmdb(fpath_net,
                     fpath_weights,
                     keys,
                     dst_prefix,
                     modes=[caffe.TRAIN, caffe.TEST],
                     ):
    """
    keys -- name of responses to extract. Must be valid for all requested modes
    """
    out = dict.fromkeys(modes)
    
    for m in modes:
        num_passes = est_min_num_fwd_passes(fpath_net, ['train', 'test'][m])
        out[m] = infer_to_lmdb(caffe.Net(fpath_net, fpath_weights, m),
                               keys,
                               num_passes,
                               dst_prefix + '%s_' + ['train', 'test'][m] + '_lmdb',
                               preserve_batch=False)
    return out

if __name__ == '__main__':
    
#     from os.path import expanduser
#          
#     fpath_net = expanduser('~/models/dark/mnist/t0/lenet_train_test_inf.prototxt')
#     fpath_weights = expanduser('~/models/dark/mnist/t0/lenet_iter_10000.caffemodel')
#     net = caffe.Net(fpath_net, fpath_weights, caffe.TRAIN)
#      
#     fpath = expanduser('~/models/dark/mnist/t0/mnist_ip2_train')
#     num_entries = read_lmdb.num_entries(expanduser('~/data/mnist/mnist_train_lmdb'))
#     
#     infer_to_h5_fixed_dims(net, ['ip2'], num_entries, "%s.h5" % fpath)
#     infer_to_lmdb(net, ['ip2'], num_entries, "%s_lmdb" % fpath)

    from os.path import expanduser
         
    fpath_net = expanduser('~/models/dark/mnist/t0/lenet_train_test.prototxt')
    fpath_weights = expanduser('~/models/dark/mnist/t0/lenet_iter_10000.caffemodel')
    
    x = response_to_lmdb(fpath_net, fpath_weights,
                     ['ip2', 'ip1'],
                     expanduser('~/models/dark/mnist/t0/mnistX_'))

    print x
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