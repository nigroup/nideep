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
                                   chunks=None
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
    chunks -- forward chunks parameter to use during hdf5 writing
    """
    if os.path.abspath(fpath) == os.path.abspath(fpath_dst):
        raise IOError("Cannot read and write to the same file (%s) (%s)" %
                      (fpath, fpath_dst))
    
    with h5py.File(fpath, 'r') as h_src:
        labls = h_src[key_label][:]
        bal = Balancer(np.squeeze(labls))
        class_count = bal.get_class_count(other_clname=other_clname)
        idxs = bal.get_idxs_to_balance_class_count(class_count.values())
        np.random.shuffle(idxs) # shuffle the array along the first index of a multi-dimensional array, in-place
        with h5py.File(fpath_dst, 'w') as h_dst:
            h_dst[key_label] = labls[idxs]
            for k in keys:
                dataset_src = h_src[k]
                shape_new = list(dataset_src.shape)
                shape_new[0] = len(idxs)
                dataset_dst = h_dst.create_dataset(k, tuple(shape_new),
                                                   dataset_src.dtype,
                                                   chunks=chunks)
                for idx_dst, idx_src in enumerate(idxs):
                    dataset_dst[idx_dst] = dataset_src[idx_src]
    return idxs

def save_balanced_sampled_class_count_hdf5(fpath,
                                           keys,
                                           fpath_dst,
                                           key_label='label',
                                           other_clname=CLNAME_OTHER,
                                           chunks=None,
                                           target_count=None
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
    chunks -- forward chunks parameter to use during hdf5 writing
    """
    if os.path.abspath(fpath) == os.path.abspath(fpath_dst):
        raise IOError("Cannot read and write to the same file (%s) (%s)" %
                      (fpath, fpath_dst))
    
    with h5py.File(fpath, 'r') as h_src:
        labls = h_src[key_label][:]
        bal = Balancer(np.squeeze(labls))
        class_count = bal.get_class_count(other_clname=other_clname)
        idxs = bal.get_idxs_to_balance_class_count(class_count.values(),
                                                   target_count)
        np.random.shuffle(idxs) # shuffle the array along the first index of a multi-dimensional array, in-place
        with h5py.File(fpath_dst, 'w') as h_dst:
            h_dst[key_label] = labls[idxs]
            for k in keys:
                dataset_src = h_src[k]
                shape_new = list(dataset_src.shape)
                shape_new[0] = len(idxs)
                dataset_dst = h_dst.create_dataset(k, tuple(shape_new),
                                                   dataset_src.dtype,
                                                   chunks=chunks)
                for idx_dst, idx_src in enumerate(idxs):
                    dataset_dst[idx_dst] = dataset_src[idx_src]
    return idxs