from nose.tools import assert_is_none
import nyudv2_to_lmdb as n2l

import os

def split_matfile_to_val_list_invalid_path_dir():
    
    n2l.split_matfile_to_val_list(os.curdir)