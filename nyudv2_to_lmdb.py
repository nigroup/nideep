'''
Created on Oct 27, 2015

@author: kashefy
'''
import os

# obtain from https://gist.github.com/kashefy/2c098bd356dea090001e#file-filesystemutils-py
import fileSystemUtils as fs
    
def split_matfile_to_val_list(fpath):
    """
    Load split file from NYU depth v2 site and
    extract list of indices that belong in validation set
    """
    if not os.path.isfile(fpath):
        raise IOError("Path is not a regular file (%s)" % fpath)

def main(args):
    
    split_path = '/home/kashefy/data/nyudv2/split.mat'
    val_list = split_matfile_to_val_list(split_path)
        
        
    return 0

if __name__ == '__main__':
    
    main(None)
    
    pass