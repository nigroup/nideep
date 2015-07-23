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

CAFFE_ROOT = '/home/kashefy/src/caffe_forks/bvlc/'
if CAFFE_ROOT is not None:
    import sys
    sys.path.insert(0,  os.path.join(CAFFE_ROOT, 'python'))
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

def get_labels_list(fpath):
    
    _, ext = os.path.splitext(fpath)
    if not ext.endswith('.txt'):
        
        raise ValueError("Invalid extension for labels list file.")
    
    with open(fpath, 'r') as f:
        
        labels_list = [line.translate(None, ''.join('\n')).split(': ')
                       for line in f if ':' in line]
    
    return labels_list

def get_labels_lut(labels_list, labels_subset):

    pairs = []
    len_labels_list = len(labels_list)
    for id_, name in labels_subset:
        
        found = False
        idx = 0
        while idx < len_labels_list and not found:
            
            id2, name2 = labels_list[idx]
            
            if name2 == name:
                pairs.append([int(id2), int(id_)])
                found = True
            
            idx += 1
            
        if not found:
            print "Could not find %s" % name
    
    #print len(labels_list)
    labels_lut = np.zeros((len(labels_list)+1,), dtype='int')
    #print pairs
    for src, dst in pairs:
        labels_lut[src] = dst
            
    return labels_lut

def pascal_context_to_lmdb(dir_imgs,
                           dir_segm_labels,
                           fpath_labels_list,
                           fpath_labels_list_subset,
                           dst_prefix,
                           dir_dst,
                           CAFFE_ROOT=None):
    '''
    Fine intersection of filename in both directories and create
    one lmdb directory for each
    '''
    if dst_prefix is None:
        dst_prefix = ''
    
    labels_list = get_labels_list(fpath_labels_list)
    labels_59_list = get_labels_list(fpath_labels_list_subset)
    
    #print labels_list
    #print labels_59_list
    labels_lut = get_labels_lut(labels_list, labels_59_list)
    #print labels_lut
    
    paths_imgs = fs.gen_paths(dir_imgs, fs.filter_is_img)
    
    paths_segm_labels = fs.gen_paths(dir_segm_labels)
     
    paths_pairs = fs.fname_pairs(paths_imgs, paths_segm_labels)    
    paths_imgs, paths_segm_labels = map(list, zip(*paths_pairs))
    
    #for a, b in paths_pairs:
    #    print a,b
    
    fpath_lmdb_imgs = os.path.join(dir_dst,
                                   '%scontext_imgs_lmdb' % dst_prefix)
    to_lmdb.imgs_to_lmdb(paths_imgs,
                         fpath_lmdb_imgs,
                         CAFFE_ROOT=CAFFE_ROOT)
    
    def apply_labels_lut(m):
        return labels_lut[m]
    
    fpath_lmdb_segm_labels = os.path.join(dir_dst,
                                          '%scontext_labels_lmdb' % dst_prefix)
    to_lmdb.matfiles_to_lmdb(paths_segm_labels,
                             fpath_lmdb_segm_labels,
                             'LabelMap',
                             CAFFE_ROOT=CAFFE_ROOT, lut=apply_labels_lut)
    
    return len(paths_imgs), fpath_lmdb_imgs, fpath_lmdb_segm_labels

def main(args):
    
    n, fpath_lmdb_imgs, fpath_lmdb_segm_labels = \
    pascal_context_to_lmdb('/media/win/Users/woodstock/dev/data/VOCdevkit/VOC2012/JPEGImagesX',
                           '/media/win/Users/woodstock/dev/data/PASCAL-Context/trainval/',
                           '/media/win/Users/woodstock/dev/data/PASCAL-Context/labels.txt',
                           '/media/win/Users/woodstock/dev/data/PASCAL-Context/59_labels.txt',
                           'X_',
                           '/media/win/Users/woodstock/dev/data/PASCAL-Context/'
                           )
    
    print "size: %d" % n, fpath_lmdb_imgs, fpath_lmdb_segm_labels
    
    #load    
    
    #with open("/media/win/Users/woodstock/dev/data/models/fcn_segm/train_val2.prototxt", 'w') as f:
    #    f.write(str(gen_net(os.path.join(dir_dst, '59_context_imgs_lmdb'), 1)))
        
    #view_segm_lmdb(2, '/media/win/Users/woodstock/dev/data/models/fcn_segm/solver2.prototxt')
        
        
    return 0

if __name__ == '__main__':
    
    main(None)
    
    pass