'''
Created on Oct 28, 2015

@author: kashefy
'''

def hwc_to_chw(m):
    '''
    Transpose 3-d matrix from H x W x C to C x H x W where C is no. of channels
    '''
    if m.ndim == 3:
        m = m.transpose((2, 0, 1))
    #elif m.ndim == 2:
    else:
        AttributeError("No. of dimensions (%d) not supported." % m.ndim)
    
    return m

def cwh_to_chw(m):
    '''
    Reorder 3-dim array from C x W x H to C x H x W
    '''
    if m.ndim == 3:
        m = m.transpose((0, 2, 1))
    else:
        AttributeError("No. of dimensions (%d) not supported." % m.ndim)
    
    return m