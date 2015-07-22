'''
Created on Jul 21, 2015

@author: kashefy
'''
import os
import numpy as np
import to_lmdb
import fileSystemUtils as fs

import cv2 as cv2
import cv2.cv as cv
import lmdb
import caffe
from caffe import layers as L
from caffe import params as P

def gen_net(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)
    return n.to_proto()

def view_segm_lmdb(nb_imgs, path_solver):
    
    solver = caffe.SGDSolver(path_solver)
     
    for _ in xrange(nb_imgs):
         
        solver.net.forward()  # train net
         
        d = solver.net.blobs['data'].data
        print d.shape
        d = np.squeeze(d, axis=(0,)) # get rid of elements dimensions
        y = cv2.cvtColor(cv2.merge([ch for ch in d]), cv.CV_RGB2BGR)
         
        #print y.dtype, y.max()
         
        cv2.imshow('data', y)
         
        d = solver.net.blobs['label'].data
        print d.shape
        d = np.squeeze(d, axis=(0,))
        
        print d
        
        cv2.waitKey()
        
    return 0

def main(args):
    
    dir_dst = '/media/win/Users/woodstock/dev/data/PASCAL-Context/'
     
    dir_imgs = '/media/win/Users/woodstock/dev/data/VOCdevkit/VOC2012/JPEGImages'
    paths_imgs = fs.gen_paths(dir_imgs, fs.filter_is_img)
    
    dir_segm_labels = '/media/win/Users/woodstock/dev/data/PASCAL-Context/trainval/trainval'
    paths_segm_labels = fs.gen_paths(dir_segm_labels)
     
    paths_pairs = fs.fname_pairs(paths_imgs, paths_segm_labels)    
    paths_imgs, paths_segm_labels = map(list, zip(*paths_pairs))
     
    #for a, b in paths_pairs:
    #    print a,b
     
    to_lmdb.imgs_to_lmdb(paths_imgs, os.path.join(dir_dst, 'context_imgs_lmdb'))
    to_lmdb.matfiles_to_lmdb(paths_segm_labels, os.path.join(dir_dst, 'context_labels_lmdb'), 'LabelMap')
    
    #load    
    
    #with open("/media/win/Users/woodstock/dev/data/models/fcn_segm/train_val2.prototxt", 'w') as f:
    #    f.write(str(gen_net(os.path.join(dir_dst, '59_context_imgs_lmdb'), 1)))
        
    #view_segm_lmdb(2, '/media/win/Users/woodstock/dev/data/models/fcn_segm/solver2.prototxt')
        
        
    return 0

if __name__ == '__main__':
    
    main(None)
    
    pass