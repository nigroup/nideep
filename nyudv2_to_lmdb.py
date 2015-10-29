'''
Created on Oct 27, 2015

@author: kashefy
'''
import os
import numpy as np
from scipy import io
import h5py
from dataset_utils import get_train_val_split_from_idx
import mat_utils as mu
import to_lmdb

class NYUDV2DataType:
    IMAGES = 'images'
    LABELS = 'labels'
    DEPTHS = 'depths'
    
def split_matfile_to_val_list(fpath):
    """
    Load split file from NYU depth v2 site and
    extract list of indices that belong in validation set
    """
    if not os.path.isfile(fpath):
        raise IOError("Path is not a regular file (%s)" % fpath)
    
    _, ext = os.path.splitext(fpath)
    
    if ext != '.mat':
        raise IOError("Invalid file type, expecting mat file (%s)" % fpath)
    
    fieldname = 'testNdxs'
    val = io.loadmat(fpath)[fieldname]
    val -= 1 # go from 1-based to zero-based indexing
    val_list = val.ravel().tolist()
    
    return val_list

def big_arr_to_arrs(a):
    """
    Turn NxCxWxH array into list of CxHxW for Caffe
    """
    return [mu.cwh_to_chw(x) for x in a]

def nyudv2_to_lmdb(path_mat,
                   dst_prefix,
                   dir_dst,
                   val_list=None):
    
    if not os.path.isfile(path_mat):
        raise IOError("Path is not a regular file (%s)" % path_mat)
    
    _, ext = os.path.splitext(path_mat)
    
    if ext != '.mat':
        raise IOError("Invalid file type, expecting mat file (%s)" % path_mat)
    
    try:
        data = io.loadmat(path_mat)
    except NotImplementedError:
        data = h5py.File(path_mat) # support version >= 7.3 matfile HDF5 format
        pass
    
    paths_lmdb = []
        
    for typ in [NYUDV2DataType.IMAGES,
                NYUDV2DataType.LABELS,
                NYUDV2DataType.DEPTHS]:
        
        typ = NYUDV2DataType.IMAGES
        
        if typ == NYUDV2DataType.IMAGES:
            
            dat = [mu.cwh_to_chw(x).astype(np.float) for x in data[typ]]
            
        elif typ == NYUDV2DataType.LABELS:
            
            dat = np.expand_dims(data[typ], axis=1).astype(int)
            
        elif typ == NYUDV2DataType.DEPTHS:
            
            dat = np.expand_dims(data[typ], axis=1).astype(np.float)
        else:
            raise ValueError("unknown NYUDV2DataType")
            
        # do train/val split
        n = len(dat)
        train_idx, val_idx = get_train_val_split_from_idx(n, val_list)
        
        nt = len(train_idx)
        nv = len(val_idx)
        
    #     # len(ndarray) same as ndarray.shape[0]
    #     if  len(labels) != len(imgs):
    #         raise ValueError("No. of images != no. of labels. (%d) != (%d)",
    #                          len(imgs), len(labels))
    #         
    #     if  len(labels) != len(depths):
    #         raise ValueError("No. of depths != no. of labels. (%d) != (%d)",
    #                          len(depths), len(labels))
        
        print len(dat), dat[0].shape
        fpath_lmdb = os.path.join(dir_dst,
                                        '%s%s_train_lmdb' % (dst_prefix, typ))
        to_lmdb.arrays_to_lmdb([dat[i] for i in train_idx], fpath_lmdb)
        
        paths_lmdb.append(fpath_lmdb)
        
        fpath_lmdb = os.path.join(dir_dst,
                                      '%s%s_val_lmdb' % (dst_prefix, typ))
        to_lmdb.arrays_to_lmdb([dat[i] for i in val_idx], fpath_lmdb)
        
        paths_lmdb.append(fpath_lmdb)
    
    print paths_lmdb

    return

    
#     # ground truth
#     print labels.shape
#     labels = big_arr_to_arrs(labels)
#     print len(labels), labels[0].shape
#     fpath_lmdb_labels_train = os.path.join(dir_dst,
#                                                 '%slabels_train_lmdb' % dst_prefix)
#     to_lmdb.arrays_to_lmdb(imgs, fpath_lmdb_labels_train)
#     
#     fpath_lmdb_labels_val = os.path.join(dir_dst,
#                                               '%slabels_val_lmdb' % dst_prefix)
#     to_lmdb.arrays_to_lmdb(imgs, fpath_lmdb_labels_val)
#     
#     # depths
#     depths = big_arr_to_arrs(depths)
#     print len(depths), depths[0].shape
#     fpath_lmdb_depths_train = os.path.join(dir_dst,
#                                                 '%slabels_train_lmdb' % dst_prefix)
#     to_lmdb.arrays_to_lmdb(imgs, fpath_lmdb_imgs_val)
#     
#     fpath_lmdb_segm_labels_val = os.path.join(dir_dst,
#                                               '%slabels_val_lmdb' % dst_prefix)
#     to_lmdb.arrays_to_lmdb(imgs, fpath_lmdb_imgs_val)
#     
#     return nt, nv, \
#         fpath_lmdb_imgs_train, fpath_lmdb_labels_train,\
#         fpath_lmdb_imgs_val, fpath_lmdb_labels_val
    
    
def main(args):
    
    split_path = '/home/kashefy/data/nyudv2/splits.mat'
    val_list = split_matfile_to_val_list(split_path)
    
    nt, nv, \
    fpath_imgs_train, fpath_labels_train, \
    fpath_imgs_val, fpath_labels_val = \
    nyudv2_to_lmdb(os.path.expanduser('~/data/nyudv2/nyu_depth_v2_labeled.mat'),
                   'nyudv2_',
                   os.path.expanduser('~/data/nyudv2'),
                   val_list=val_list
                   )
    
    print "size: %d" % (nt+nv), fpath_imgs_train, fpath_labels_train
    
        
        
    return 0

if __name__ == '__main__':
    
    main(None)
    
    pass