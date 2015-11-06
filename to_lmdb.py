'''
Created on Jul 18, 2015

@author: kashefy
'''
import os
import numpy as np
from scipy import io
import lmdb
from read_img import read_img_cv2
import caffe

NUM_IDX_DIGITS = 10
IDX_FMT = '{:0>%d' % NUM_IDX_DIGITS + 'd}'

def imgs_to_lmdb(paths_src, path_dst):
    '''
    Generate LMDB file from set of images
    Source: https://github.com/BVLC/caffe/issues/1698#issuecomment-70211045
    credit: Evan Shelhamer
    '''
    
    db = lmdb.open(path_dst, map_size=int(1e12))
    
    with db.begin(write=True) as in_txn:
    
        for idx, path_ in enumerate(paths_src):
            
            img = read_img_cv2(path_)
            img_dat = caffe.io.array_to_datum(img)
            in_txn.put(IDX_FMT.format(idx), img_dat.SerializeToString())
    
    db.close()

    return 0

def matfiles_to_lmdb(paths_src, path_dst, fieldname,
                     lut=None):
    '''
    Generate LMDB file from set of mat files with integer data
    Source: https://github.com/BVLC/caffe/issues/1698#issuecomment-70211045
    credit: Evan Shelhamer
    
    '''    
    db = lmdb.open(path_dst, map_size=int(1e12))
    
    with db.begin(write=True) as in_txn:
    
        for idx, path_ in enumerate(paths_src):
            
            content_field = io.loadmat(path_)[fieldname]
            # get shape (1,H,W)
            while len(content_field.shape) < 3:
                content_field = np.expand_dims(content_field, axis=0)
            content_field = content_field.astype(int)
            
            if lut is not None:
                content_field = lut(content_field)
                
            img_dat = caffe.io.array_to_datum(content_field)
            in_txn.put(IDX_FMT.format(idx), img_dat.SerializeToString())
    
    db.close()

    return 0

def scalars_to_lmdb(scalars, path_dst,
                    lut=None):
    '''
    Generate LMDB file from list of scalars
    '''    
    db = lmdb.open(path_dst, map_size=int(1e12))
    
    with db.begin(write=True) as in_txn:
    
        for idx, x in enumerate(scalars):
            
            content_field = np.array([x])
            # get shape (1,1,1)
            while len(content_field.shape) < 3:
                content_field = np.expand_dims(content_field, axis=0)
            content_field = content_field.astype(int)
            
            if lut is not None:
                content_field = lut(content_field)
                
            dat = caffe.io.array_to_datum(content_field)
            in_txn.put(IDX_FMT.format(idx), dat.SerializeToString())
    
    db.close()

    return 0

def arrays_to_lmdb(arrs, path_dst):
    '''
    Generate LMDB file from list of ndarrays    
    '''
    db = lmdb.open(path_dst, map_size=int(1e12))
    
    with db.begin(write=True) as in_txn:
    
        for idx, x in enumerate(arrs):
            
            content_field = x
            while len(content_field.shape) < 3:
                content_field = np.expand_dims(content_field, axis=0)
            
            dat = caffe.io.array_to_datum(content_field)
            in_txn.put(IDX_FMT.format(idx), dat.SerializeToString())
    
    db.close()

    return 0

