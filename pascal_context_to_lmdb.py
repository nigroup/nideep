'''
Created on Jul 21, 2015

@author: kashefy
'''
import os
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
        print solver.net.blobs['data'].data.shape
         
        d = solver.net.blobs['data'].data
        sh = d.shape
        d = d.reshape(sh[1], sh[2], sh[3])
        y = cv2.cvtColor(cv2.merge([d[0, :, :], d[1, :, :], d[2, :, :]]), cv.CV_RGB2BGR)
         
        print y.dtype, y.max()
         
        cv2.imshow('data', y)
         
        d = solver.net.blobs['label'].data
        sh = d.shape
        d = d.reshape(sh[1], sh[2], sh[3])
        y = cv2.cvtColor(cv2.merge([d[0, :, :], d[1, :, :], d[2, :, :]]), cv.CV_RGB2BGR)
         
        print y.dtype, y.max()
         
        cv2.imshow('label', y)
        cv2.waitKey()
        
    return 0

def main(args):
    
    dir_dst = '/media/win/Users/woodstock/dev/data/PASCAL-Context/'
     
    dir_imgs = '/media/win/Users/woodstock/dev/data/VOCdevkit/VOC2012/JPEGImagesX'
    paths_imgs = fs.gen_paths(dir_imgs, fs.filter_is_img)
     
    dir_segm_labels = '/media/win/Users/woodstock/dev/data/PASCAL-Context/59_context_labels'
    paths_segm_labels = fs.gen_paths(dir_segm_labels, fs.filter_is_img)
    
    dir_segm_labels = '/media/win/Users/woodstock/dev/data/PASCAL-Context/trainval/trainval'
    paths_segm_labels = fs.gen_paths(dir_segm_labels)
     
    paths_pairs = fs.fname_pairs(paths_imgs, paths_segm_labels)    
    paths_imgs, paths_segm_labels = map(list, zip(*paths_pairs))
     
    for a, b in paths_pairs:
        print a,b
     
    to_lmdb.imgs_to_lmdb(paths_imgs, os.path.join(dir_dst, 'context_imgs_lmdb'))
    to_lmdb.matfiles_to_lmdb(paths_segm_labels, os.path.join(dir_dst, 'context_labels_lmdb'), 'LabelMap')
    
    #load    
    
    #with open("/media/win/Users/woodstock/dev/data/models/fcn_segm/train_val2.prototxt", 'w') as f:
    #    f.write(str(gen_net(os.path.join(dir_dst, '59_context_imgs_lmdb'), 1)))
        
    #view_segm_lmdb(20, '/media/win/Users/woodstock/dev/data/models/fcn_segm/solver2.prototxt')
        
        
    return 0

if __name__ == '__main__':
    
    main(None)
    
    pass