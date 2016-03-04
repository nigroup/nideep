'''
Created on Mar 3, 2016

@author: kashefy
'''
import numpy as np
import h5py
from astropy.wcs.docstrings import delta

def get_class_count_hdf5(fpath, key='label', other_clname='other'):
    
    f = h5py.File(fpath, 'r')
    l = np.squeeze(f[key])
    b = Balancer(l)
    x = b.get_class_count(key=key, other_clname=other_clname)
    print x.values()
    b.balance_class_count(x.values())
    return x

class Balancer(object):
    '''
    Balance class counts
    '''
    def get_class_count(self, key='label', other_clname='other'):
        
        if self.l.ndim == 2:
            class_count = np.sum(self.l, axis=0)
            other_count = int(len(self.l) - np.sum(class_count))
        d = {}
        for i, x in enumerate(class_count):
            d[i] = int(x)
        if other_clname is not None and other_clname != '':
            d[other_clname] = other_count
        return d
    
    def balance_class_count(self, class_counts):
    
        print class_counts, self.l.shape
        mx = np.max(class_counts)
        
        for i, c in enumerate(class_counts):
            
            
            delta = mx - c
            if delta > 0:
                rows = np.where(self.l[:, i]==1)[0]
                print rows.size
                rows_sampled = rows[np.random.randint(0, high=rows.size, size=(delta, 1))]
                
            #    if delta > c:
        
        print class_counts, mx

    def __init__(self, labls):
        '''
        Constructor
        labls -- ground truth
        '''
        self.l = labls
        