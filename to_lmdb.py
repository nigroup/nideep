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
import caffe
from caffe import layers as L
from caffe import params as P

def imgs_to_lmdb(paths_src, path_dst):
    '''
    Generate LMDB file from set of images
    Source: https://github.com/BVLC/caffe/issues/1698#issuecomment-70211045
    credit: Evan Shelhamer
    
    '''
    db = lmdb.open(path_dst, map_size=int(1e12))
    
    with db.begin(write=True) as in_txn:
    
        for idx, path_ in enumerate(paths_src):
            # load image:
            # - as np.uint8 {0, ..., 255}
            # - in BGR (switch from RGB)
            # - in Channel x Height x Width order (switch from H x W x C)
            # or load whatever ndarray you need
            img = cv2.imread(path_)
            print "img.shape", img.shape, img.max()
            img = img[:, :, ::-1]
            img = img.transpose((2, 0, 1))
            print "after", img.shape
            
            img_dat = caffe.io.array_to_datum(img)
            in_txn.put('{:0>10d}'.format(idx), img_dat.SerializeToString())
    
    db.close()

    return 0

def matfiles_to_lmdb(paths_src, path_dst, fieldname):
    '''
    Generate LMDB file from set of images
    Source: https://github.com/BVLC/caffe/issues/1698#issuecomment-70211045
    credit: Evan Shelhamer
    
    '''
    db = lmdb.open(path_dst, map_size=int(1e12))
    
    with db.begin(write=True) as in_txn:
    
        for idx, path_ in enumerate(paths_src):
            
            content_field = io.loadmat(path_)[fieldname]
            content_field = np.expand_dims(content_field, axis=0)
            content_field = content_field.astype(int)
            #print content_field.shape
            
            img_dat = caffe.io.array_to_datum(content_field)
            in_txn.put('{:0>10d}'.format(idx), img_dat.SerializeToString())
    
    db.close()

    return 0

def gen_net(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)
    return n.to_proto()

def main(args):
    
    paths_in = ['/media/win/Users/woodstock/dev/data/lena.png']
    
    caffe.set_mode_cpu()
    
    path_lmdb_train = 'image-lmdb'
    path_lmdb_test = path_lmdb_train
    
    imgs_to_lmdb(paths_in, path_lmdb_train)
    
    with open('train.prototxt', 'w') as f:
        f.write(str(gen_net(path_lmdb_train, 1)))
    
    with open('test.prototxt', 'w') as f:
        f.write(str(gen_net(path_lmdb_test, 1)))
    
    solver = caffe.SGDSolver('auto_solver.prototxt')
    
    for i in xrange(len(paths_in)):
        
        solver.net.forward()  # train net
        print solver.net.blobs['data'].data.shape
        
        d = solver.net.blobs['data'].data
        sh = d.shape
        d = d.reshape(sh[1], sh[2], sh[3])
        y = cv2.merge([d[0, :, :], d[1, :, :], d[2, :, :]])
        print y.dtype
        print y
        cv2.imshow('y', y)
        cv2.waitKey()
    
    return 0

if __name__ == '__main__':
    
    main(None)
    
    pass