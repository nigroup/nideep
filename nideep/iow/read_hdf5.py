'''
Created on Apr 6, 2016

@author: marcenacp
'''
import h5py

def num_entries(path_hdf5, key_label):
    """
    Get no. of entries in hdf5
    """
    num_entries = 0
    with open(path_hdf5, 'r') as f:
        for db_hdf5 in f:
            with h5py.File(db_hdf5.rstrip()) as db:
                num_entries += db[key_label].shape[0]
    f.close()
    return num_entries
