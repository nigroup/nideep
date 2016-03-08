'''
Created on Mar 7, 2016

@author: kashefy
'''
import os
import numpy as np
import h5py
from balance import Balancer, CLNAME_OTHER

def get_class_count_hdf5(fpath,
                         key_label='label',
                         other_clname=CLNAME_OTHER):
    """ Count per-class instances in HDF5 and return a dictionary of class ids
    and per-class count
    
    fpath -- path to HDF5 file
    Keyword arguments:
    key_label -- key for ground truth data in HDF5
    other_clname -- name for negative class (None if non-existent)
    """
    h = h5py.File(fpath, 'r')
    b = Balancer(np.squeeze(h[key_label]))
    return b.get_class_count(other_clname=other_clname)

def balance_class_count_hdf5(fpath, keys,
                             key_label='label',
                             other_clname=CLNAME_OTHER):
    """ Resample keys in an HDF5 to generate a near balanced dataset.
    Returns a dictionary with resampled features and ground truth
    and indicies from the original label that were sampled.
    Not suitable for very large datasets.
    
    fpath -- path to HDF5 file
    keys -- keys to resample (e.g. features)
    Keyword arguments:
    key_label -- key for ground truth data in HDF5
    other_clname -- name for negative class (None if non-existent)
    """
    h_src = h5py.File(fpath, 'r')
    labls = h_src[key_label][:]
    bal = Balancer(np.squeeze(labls))
    class_count = bal.get_class_count(other_clname=other_clname)
    idxs = bal.get_idxs_to_balance_class_count(class_count.values())
    np.random.shuffle(idxs) # shuffle the array along the first index of a multi-dimensional array, in-place
    
    dict_balanced = {key_label : labls[idxs]}
    for k in keys:
        dict_balanced[k] = h_src[k][:][idxs]
    return dict_balanced, idxs

def save_balanced_class_count_hdf5(fpath,
                                   keys,
                                   fpath_dst,
                                   key_label='label',
                                   other_clname=CLNAME_OTHER,
                                   ):
    """ Resample keys in an HDF5 to generate a near balanced dataset
    and save into a new HDF5.
    Returns indicies from the original label that were sampled.
    Not suitable for very large datasets.
    
    fpath -- path to source HDF5 file
    keys -- keys to resample (e.g. features)
    fpath_dst -- path to destination HDF5 file
    Keyword arguments:
    key_label -- key for ground truth data in HDF5
    other_clname -- name for negative class (None if non-existent)
    """
    if os.path.abspath(fpath) == os.path.abspath(fpath_dst):
        raise IOError("Cannot read and write to the same file (%s) (%s)" %
                      (fpath, fpath_dst))
    
    h_src = h5py.File(fpath, 'r')
    labls = h_src[key_label][:]
    bal = Balancer(np.squeeze(labls))
    class_count = bal.get_class_count(other_clname=other_clname)
    idxs = bal.get_idxs_to_balance_class_count(class_count.values())
    np.random.shuffle(idxs) # shuffle the array along the first index of a multi-dimensional array, in-place
    
    h_dst = h5py.File(fpath_dst, 'w')
    h_dst[key_label] = labls[idxs]
    for k in keys:
        shape_new = list(h_src[k].shape)
        shape_new[0] = len(idxs)
        dset = h_dst.create_dataset(k, tuple(shape_new), h_src[k].dtype)
        for idx_dst, idx_src in enumerate(idxs):
            dset[idx_dst] = h_src[k][idx_src]
    h_src.close()
    h_dst.close()
    return idxs