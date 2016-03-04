'''
Created on Mar 3, 2016

@author: kashefy
'''
import numpy as np
import h5py

def get_class_count_hdf5(fpath, key_label='label', other_clname='other'):
    
    h = h5py.File(fpath, 'r')
    l = np.squeeze(h[key_label])
    b = Balancer(l)
    return b.get_class_count(other_clname=other_clname)

def balance_class_count_hdf5(fpath, key_feat, key_label='label', other_clname='other'):
    
    h = h5py.File(fpath, 'r')
    labls = np.squeeze(h[key_label])
    bal = Balancer(np.squeeze(labls))
    class_count = bal.get_class_count(other_clname=other_clname)
    idxs = bal.get_idxs_to_balance_class_count(class_count.values())
    idxs = np.random.shuffle(idxs) # this only shuffles the array along the first index of a multi-dimensional array
    return h[key_feat][idxs], labls[idxs]

def balance_class_count(self, feats, labls, other_clname='other'):
    
    bal = Balancer(labls)
    class_count = bal.get_class_count(other_clname=other_clname)
    idxs = bal.get_idxs_to_balance_class_count(class_count.values())
    idxs = np.random.shuffle(idxs) # this only shuffles the array along the first index of a multi-dimensional array
    return feats[idxs], labls[idxs]
    
class Balancer(object):
    '''
    Balance class counts
    '''
    def get_class_count(self, other_clname='other'):
        
        self.has_other_cl = other_clname is not None and other_clname != ''
        
        if self.l.ndim == 2:
            class_count = np.sum(self.l, axis=0)
            other_count = int(len(self.l) - np.sum(class_count))
        d = {}
        for i, x in enumerate(class_count):
            d[i] = int(x)
        if self.has_other_cl:
            d[other_clname] = other_count
        return d
    
    def get_idxs_to_balance_class_count(self, class_counts):
    
        mx = np.max(class_counts)
        idxs = np.arange(self.l.size).reshape(-1, 1)
        for i, c in enumerate(class_counts):
            delta_ = mx - c
            if delta_ > 0:
                rows = np.where(self.l[:, i]==1)[0]
                rows_to_sample = rows[np.random.randint(0, high=rows.size,
                                                        size=(delta_, 1))
                                      ]
                np.vstack((idxs, rows_to_sample))
        #if has_other_cl:
        # TODO
        return idxs

    def __init__(self, labls):
        '''
        Constructor
        labls -- ground truth
        '''
        self.l = labls
        self.has_other_cl = True
        