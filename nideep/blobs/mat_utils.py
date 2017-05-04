'''
Created on Oct 28, 2015

@author: kashefy
'''
import numpy as np

def hwc_to_chw(m):
    '''
    Transpose 3-d matrix from H x W x C to C x H x W where C is no. of channels
    '''
    if m.ndim == 3:
        m = m.transpose((2, 0, 1))
    # elif m.ndim == 2:
    else:
        raise AttributeError("No. of dimensions (%d) not supported." % m.ndim)

    return m

def cwh_to_chw(m):
    '''
    Reorder 3-dim array from C x W x H to C x H x W
    '''
    if m.ndim == 3:
        m = m.transpose((0, 2, 1))
    else:
        raise AttributeError("No. of dimensions (%d) not supported." % m.ndim)

    return m

def expand_dims(m, d):
    '''
    Expand dimensions in-place starting from first axis (axis=0) until we reach d dims.
    '''
    while m.ndim < d:
        m = np.expand_dims(m, axis=0)
    return m
