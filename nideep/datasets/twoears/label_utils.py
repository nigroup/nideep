'''
Created on Apr 25, 2017

@author: kashefy
'''
import numpy as np
import h5py

def id_loc_to_loc(fpath_src, key_dst, key_src='label_id_loc', has_void_bin=True):
    
    with h5py.File(fpath_src, 'r+') as h:
        if has_void_bin:
            l = np.sum(h[key_src][...,:-1], axis=1)
        else:
            l = np.sum(h[key_src], axis=1)
        l = np.expand_dims(l, 1)
        h[key_dst] = l
if __name__ == '__main__':
    pass