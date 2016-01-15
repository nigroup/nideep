'''
Created on Jan 15, 2016

@author: kashefy
'''
import h5py
from mat_utils import expand_dims

def arrays_to_h5_fixed(arrs, key, path_dst):
    """
    """        
    with h5py.File(path_dst, "w") as f:
        
        for x in arrs:
            content_field = expand_dims(x, 3)
            f[key] = content_field
            