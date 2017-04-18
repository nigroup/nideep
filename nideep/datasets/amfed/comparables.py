'''
Created on Mar 29, 2017

@author: kashefy
'''
import os

def list_comparables(items, suffix=None):
    if suffix is None:
        return [ComparableEntity(x, idx) for idx, x in enumerate(items)]
    else:
        return [ComparableEntityWithSuffix(x, idx, suffix) for idx, x in enumerate(items)]

class ComparableEntity(object):
    '''
    for comparing video filesnames with other entity data (e.g. au labels)
    '''
    def __init__(self, p, idx):
        self.p = p
        self.b, self.ext = os.path.splitext(os.path.basename(p))
        self.idx = idx                  
    def __eq__(self, other):
        return self.b.startswith(other.b)
    def __hash__(self):
        return hash(self.b)
    def __str__(self):
        return self.b
    
class ComparableEntityWithSuffix(ComparableEntity):
    '''
    for comparing au labels and landmarks file names with other entities (e.g. video files)
    '''
    def __init__(self, p, idx, suffix):
        super(ComparableEntityWithSuffix, self).__init__(p, idx)
        if suffix in self.b:
            self.b = self.b[:self.b.index(suffix)]