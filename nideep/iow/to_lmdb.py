'''
Created on Jul 18, 2015

@author: kashefy
'''
import numpy as np
from scipy import io
import lmdb
from read_img import read_img_cv2
import caffe

from lmdb_utils import IDX_FMT, MAP_SZ
from nideep.blobs.mat_utils import expand_dims

def imgs_to_lmdb(paths_src, path_dst):
    '''
    Generate LMDB file from set of images
    Source: https://github.com/BVLC/caffe/issues/1698#issuecomment-70211045
    credit: Evan Shelhamer
    '''

    db = lmdb.open(path_dst, map_size=MAP_SZ)

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
    db = lmdb.open(path_dst, map_size=MAP_SZ)

    with db.begin(write=True) as in_txn:

        for idx, path_ in enumerate(paths_src):

            content_field = io.loadmat(path_)[fieldname]
            # get shape (1,H,W)
            content_field = expand_dims(content_field, 3)
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
    db = lmdb.open(path_dst, map_size=MAP_SZ)

    with db.begin(write=True) as in_txn:

        if not hasattr(scalars, '__iter__'):
            scalars = np.array([scalars])

        for idx, x in enumerate(scalars):

            if not hasattr(x, '__iter__'):
                content_field = np.array([x])
            else:
                content_field = np.array(x)

            # validate these are scalars
            if content_field.size != 1:
                raise AttributeError("Unexpected shape for scalar at i=%d (%s)"
                                     % (idx, str(content_field.shape)))

            # guarantee shape (1,1,1)
            content_field = expand_dims(content_field, 3)
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
    db = lmdb.open(path_dst, map_size=MAP_SZ)

    with db.begin(write=True) as in_txn:

        for idx, x in enumerate(arrs):
            content_field = expand_dims(x, 3)

            dat = caffe.io.array_to_datum(content_field)
            in_txn.put(IDX_FMT.format(idx), dat.SerializeToString())

    db.close()
    return 0

