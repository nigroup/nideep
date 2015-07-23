'''
Created on Jul 18, 2015

@author: kashefy
'''
import os
import numpy as np
from scipy import io
import cv2 as cv2
import cv2.cv as cv
import lmdb

def imgs_to_lmdb(paths_src, path_dst, CAFFE_ROOT=None):
    '''
    Generate LMDB file from set of images
    Source: https://github.com/BVLC/caffe/issues/1698#issuecomment-70211045
    credit: Evan Shelhamer
    
    '''
    if CAFFE_ROOT is not None:
        import sys
        sys.path.insert(0,  os.path.join(CAFFE_ROOT, 'python'))
    import caffe
    
    db = lmdb.open(path_dst, map_size=int(1e12))
    
    with db.begin(write=True) as in_txn:
    
        for idx, path_ in enumerate(paths_src):
            # load image:
            # - as np.uint8 {0, ..., 255}
            # - in BGR (switch from RGB)
            # - in Channel x Height x Width order (switch from H x W x C)
            # or load whatever ndarray you need
            img = cv2.imread(path_)
            #print "img.shape", img.shape, img.max()
            img = img[:, :, ::-1]
            img = img.transpose((2, 0, 1))
            #print "after", img.shape
            
            img_dat = caffe.io.array_to_datum(img)
            in_txn.put('{:0>10d}'.format(idx), img_dat.SerializeToString())
    
    db.close()

    return 0

def matfiles_to_lmdb(paths_src, path_dst, fieldname,
                     CAFFE_ROOT=None,
                     lut=None):
    '''
    Generate LMDB file from set of images
    Source: https://github.com/BVLC/caffe/issues/1698#issuecomment-70211045
    credit: Evan Shelhamer
    
    '''
    if CAFFE_ROOT is not None:
        import sys
        sys.path.insert(0,  os.path.join(CAFFE_ROOT, 'python'))
    import caffe
    
    db = lmdb.open(path_dst, map_size=int(1e12))
    
    with db.begin(write=True) as in_txn:
    
        for idx, path_ in enumerate(paths_src):
            
            content_field = io.loadmat(path_)[fieldname]
            content_field = np.expand_dims(content_field, axis=0)
            content_field = content_field.astype(int)
            #print content_field.shape
            print 'before', content_field
            if lut is not None:
                content_field = lut(content_field)
            
            print 'after', content_field
            img_dat = caffe.io.array_to_datum(content_field)
            in_txn.put('{:0>10d}'.format(idx), img_dat.SerializeToString())
    
    db.close()

    return 0

def gen_net(lmdb, batch_size, CAFFE_ROOT):
    '''
    For lmdb data inspection
    '''
    if CAFFE_ROOT is not None:
        os.chdir(CAFFE_ROOT)
        import sys
        sys.path.insert(0, './python')
    import caffe
    from caffe import layers as L
    from caffe import params as P
    
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)
    return n.to_proto()
