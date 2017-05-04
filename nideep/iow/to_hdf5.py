'''
Created on Jan 15, 2016

@author: kashefy
'''
import os
import numpy as np
import h5py
from nideep.blobs.mat_utils import expand_dims

def arrays_to_h5_fixed(arrs, key, path_dst):
    '''
    save list of arrays (all same size) to hdf5 under a single key
    '''
    with h5py.File(path_dst, 'w') as f:
        f[key] = [expand_dims(x, 3) for x in arrs]

def split_hdf5(fpath_src, dir_dst, tot_floats=(20 * 1024 * 1024)):

    if not os.path.isdir(dir_dst):
        raise(IOError, "%s is not a directory." % (dir_dst,))

    name_, ext = os.path.splitext(os.path.basename(fpath_src))

    dst_paths = []

    with h5py.File(fpath_src, 'r') as h_src:
        keys = h_src.keys()

        # determine largest chunk size
        argmax_shape = np.argmax(np.prod([h_src[key].shape for key in keys], axis=1))
        max_shape = h_src[keys[argmax_shape]].shape

        split_sz = int(tot_floats / np.prod(max_shape[1:]))  # largest no. of elements per split

        split_count = 0
        num_saved = 0

        while num_saved < max_shape[0]:

            fpath_dst = os.path.join(dir_dst, '%s_%03d%s' % (name_,
                                                             split_count,
                                                             ext))
            with h5py.File(fpath_dst, ['a', 'w'][num_saved == 0]) as h_dst:
                for key in keys:
                    h_dst[key] = h_src[key][num_saved:num_saved + split_sz]
                dst_paths.append(fpath_dst)
            num_saved += split_sz
            split_count += 1

        return dst_paths
