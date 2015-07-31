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

def get_train_val_split(src, val_list):
    
    train_idx = []
    val_idx = []
    
    len_ = len(val_list)
    
    for i, x in enumerate(src):
        
        found = False
        j = 0
        while j < len_ and not found:
            
            found = val_list[j] in x            
            j += 1
            
        if found:
            val_idx.append(i)
        else:
            train_idx.append(i)
    
    return train_idx, val_idx

def pascal_context_to_lmdb(dir_imgs,
                           dir_segm_labels,
                           fpath_labels_list,
                           fpath_labels_list_subset,
                           dst_prefix,
                           dir_dst,
                           CAFFE_ROOT=None,
                           val_list=None):
    '''
    Fine intersection of filename in both directories and create
    one lmdb directory for each
    val_list - list of entities to exclude from train (validation subset e.g. ['2008_000002', '2010_000433'])
    '''
    if dst_prefix is None:
        dst_prefix = ''
        
    labels_list = get_labels_list(fpath_labels_list)
    labels_59_list = get_labels_list(fpath_labels_list_subset)
    
    #print labels_list
    #print labels_59_list
    labels_lut = get_labels_lut(labels_list, labels_59_list)
    def apply_labels_lut(m):
        return labels_lut[m]
    
    paths_imgs = fs.gen_paths(dir_imgs, fs.filter_is_img)
    
    paths_segm_labels = fs.gen_paths(dir_segm_labels)
     
    paths_pairs = fs.fname_pairs(paths_imgs, paths_segm_labels)    
    paths_imgs, paths_segm_labels = map(list, zip(*paths_pairs))
    
    #for a, b in paths_pairs:
    #    print a,b
    
    if val_list is not None:
        # do train/val split
        
        train_idx, val_idx = get_train_val_split(paths_imgs, val_list)
        
        # images
        paths_imgs_train = [paths_imgs[i] for i in train_idx]
        fpath_lmdb_imgs_train = os.path.join(dir_dst,
                                       '%scontext_imgs_train_lmdb' % dst_prefix)
        to_lmdb.imgs_to_lmdb(paths_imgs_train,
                             fpath_lmdb_imgs_train,
                             CAFFE_ROOT=CAFFE_ROOT)
        
        paths_imgs_val = [paths_imgs[i] for i in val_idx]
        fpath_lmdb_imgs_val = os.path.join(dir_dst,
                                       '%scontext_imgs_val_lmdb' % dst_prefix)
        to_lmdb.imgs_to_lmdb(paths_imgs_val,
                             fpath_lmdb_imgs_val,
                             CAFFE_ROOT=CAFFE_ROOT)
        
        # ground truth
        paths_segm_labels_train = [paths_segm_labels[i] for i in train_idx]
        fpath_lmdb_segm_labels_train = os.path.join(dir_dst,
                                              '%scontext_labels_train_lmdb' % dst_prefix)
        to_lmdb.matfiles_to_lmdb(paths_segm_labels_train,
                                 fpath_lmdb_segm_labels_train,
                                 'LabelMap',
                                 CAFFE_ROOT=CAFFE_ROOT, lut=apply_labels_lut)
        
        paths_segm_labels_val = [paths_segm_labels[i] for i in val_idx]
        fpath_lmdb_segm_labels_val = os.path.join(dir_dst,
                                              '%scontext_labels_val_lmdb' % dst_prefix)
        to_lmdb.matfiles_to_lmdb(paths_segm_labels_val,
                                 fpath_lmdb_segm_labels_val,
                                 'LabelMap',
                                 CAFFE_ROOT=CAFFE_ROOT, lut=apply_labels_lut)
        
        return len(paths_imgs_train), len(paths_imgs_val),\
            fpath_lmdb_imgs_train, fpath_lmdb_segm_labels_train, fpath_lmdb_imgs_val, fpath_lmdb_segm_labels_val
        
    else:
        fpath_lmdb_imgs = os.path.join(dir_dst,
                                       '%scontext_imgs_lmdb' % dst_prefix)
        to_lmdb.imgs_to_lmdb(paths_imgs,
                             fpath_lmdb_imgs,
                             CAFFE_ROOT=CAFFE_ROOT)
        
        fpath_lmdb_segm_labels = os.path.join(dir_dst,
                                              '%scontext_labels_lmdb' % dst_prefix)
        to_lmdb.matfiles_to_lmdb(paths_segm_labels,
                                 fpath_lmdb_segm_labels,
                                 'LabelMap',
                                 CAFFE_ROOT=CAFFE_ROOT, lut=apply_labels_lut)
        
        return len(paths_imgs), fpath_lmdb_imgs, fpath_lmdb_segm_labels

def main(args):
    
    val_list_path = '/home/kashefy/data/PASCAL-Context/val_59.txt'
    with open(val_list_path, 'r') as f:
        val_list = f.readlines()
        val_list = [l.translate(None, ''.join('\n')) for l in val_list if len(l) > 0]
    
    nt, nv, fpath_imgs_train, fpath_labels_train, fpath_imgs_val, fpath_labels_val = \
    pascal_context_to_lmdb('/home/kashefy/data/VOCdevkit/VOC2012/JPEGImagesX',
                           '/home/kashefy/data/PASCAL-Context/trainval/',
                           '/home/kashefy/data/PASCAL-Context/labels.txt',
                           '/home/kashefy/data/PASCAL-Context/59_labels.txt',
                           '',
                           '/home/kashefy/data/PASCAL-Context/',
                           val_list=val_list
                           )
    
    print "size: %d" % nt, nv, fpath_imgs_train, fpath_labels_train, fpath_imgs_val, fpath_labels_val
    
    #load    
    
    #with open("/media/win/Users/woodstock/dev/data/models/fcn_segm/train_val2.prototxt", 'w') as f:
    #    f.write(str(gen_net(os.path.join(dir_dst, '59_context_imgs_lmdb'), 1)))
        
    #view_segm_lmdb(2, '/media/win/Users/woodstock/dev/data/models/fcn_segm/solver2.prototxt')
        
        
    return 0

if __name__ == '__main__':
    
    main(None)
    
    pass