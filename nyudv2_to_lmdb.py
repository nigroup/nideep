'''
Created on Oct 27, 2015

@author: kashefy
'''
import os
from scipy import io
    
def split_matfile_to_val_list(fpath):
    """
    Load split file from NYU depth v2 site and
    extract list of indices that belong in validation set
    """
    if not os.path.isfile(fpath):
        raise IOError("Path is not a regular file (%s)" % fpath)
    
    _, ext = os.path.splitext(fpath)
    
    if ext != '.mat':
        raise IOError("Invalid file type, expecting mat file (%s)" % fpath)
    
    fieldname = 'testNdxs'
    val = io.loadmat(fpath)[fieldname]
    val_list = val.ravel().tolist()
    
    return val_list

def main(args):
    
    split_path = '/home/kashefy/data/nyudv2/splits.mat'
    val_list = split_matfile_to_val_list(split_path)
    
    print val_list
        
        
    return 0

if __name__ == '__main__':
    
    main(None)
    
    pass