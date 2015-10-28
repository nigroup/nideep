'''
Created on Oct 28, 2015

@author: kashefy
'''

def whc_to_chw(m):
    '''
    Reorder multi-channel image matrix from W x H x C to C x H x W
    '''
    if m.ndim == 3:
        m = m.transpose((2, 0, 1))
    #elif m.ndim == 2:
    else:
        AttributeError("No. of dimensions (%d) not supported." % m.ndim)
    
    return m

