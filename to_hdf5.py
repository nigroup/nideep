'''
Created on Jan 15, 2016

@author: kashefy
'''
import h5py
from mat_utils import expand_dims

def arrays_to_h5_fixed(arrs, key, path_dst):
    '''
    save list of arrays (all same size) to hdf5 under a single key
    '''
    with h5py.File(path_dst, "w") as f:
        f[key] = [expand_dims(x, 3) for x in arrs]
            