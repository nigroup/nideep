'''
Created on May 30, 2016

@author: kashefy
'''
import lmdb

from lmdb_utils import IDX_FMT, MAP_SZ

def copy_samples_lmdb(path_lmdb, path_dst, keys, func_data=None):
    """
    Copy select samples from an lmdb into another.
    Can be used for sampling from an lmdb into another and generating a random shuffle
    of lmdb content.
    
    Parameters:
    path_lmdb -- source lmdb
    path_dst -- destination lmdb
    keys -- list of keys or indices to sample from source lmdb
    """
    db = lmdb.open(path_dst, map_size=MAP_SZ)
    key_dst = 0
    with db.begin(write=True) as txn_dst:
        with lmdb.open(path_lmdb, readonly=True).begin() as txn_src:
             
            for key_src in keys:
                if not isinstance(key_src, basestring):
                    key_src = IDX_FMT.format(key_src)
                if func_data is None:
                    txn_dst.put(IDX_FMT.format(key_dst), txn_src.get(key_src))
                else:
                    txn_dst.put(IDX_FMT.format(key_dst), func_data(txn_src.get(key_src)))
                key_dst += 1
    db.close()
    
def concatenate_lmdb(paths_lmdb, path_dst):
    """
    Copy select samples from an lmdb into another.
    Can be used for sampling from an lmdb into another and generating a random shuffle
    of lmdb content.
    
    Parameters:
    paths_lmdb -- list of lmdbs to conatenate
    path_dst -- destination lmdb
    keys -- list of keys or indices to sample from source lmdb
    """
    db = lmdb.open(path_dst, map_size=MAP_SZ)
    key_dst = 0
    with db.begin(write=True) as txn_dst:
        for p in paths_lmdb:
            with lmdb.open(p, readonly=True).begin() as txn_src:
                for _, value in txn_src.cursor():
                    txn_dst.put(IDX_FMT.format(key_dst), value)
                    key_dst += 1
    db.close()



