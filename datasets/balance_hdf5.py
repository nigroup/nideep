'''
Created on Mar 7, 2016

@author: kashefy
'''
import numpy as np
import h5py
from balance import Balancer, CLNAME_OTHER

def get_class_count_hdf5(fpath, key_label='label', other_clname=CLNAME_OTHER):
    
    h = h5py.File(fpath, 'r')
    l = np.squeeze(h[key_label])
    b = Balancer(l)
    return b.get_class_count(other_clname=other_clname)

def balance_class_count_hdf5(fpath, keys, key_label='label', other_clname=CLNAME_OTHER):
    
    h_src = h5py.File(fpath, 'r')
    labls = np.squeeze(h_src[key_label])
    bal = Balancer(np.squeeze(labls))
    class_count = bal.get_class_count(other_clname=other_clname)
    idxs = bal.get_idxs_to_balance_class_count(class_count.values())
    np.random.shuffle(idxs) # shuffle the array along the first index of a multi-dimensional array, in-place
    
    dict_balanced = {key_label : h_src[key_label][:][idxs]}
    for k in keys:
        dict_balanced[k] = h_src[k][:][idxs]
    return dict_balanced, idxs

