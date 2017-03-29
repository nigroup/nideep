'''
Created on Jul 21, 2015

@author: kashefy
'''
import os
import nideep.datasets.dataset_utils as du
import nideep.iow.file_system_utils as fs

def get_labels_list(fpath):
    """
    Read class names from text file
    """
    _, ext = os.path.splitext(fpath)
    if not ext.endswith('.txt'):

        raise ValueError("Invalid extension for labels list file.")

    with open(fpath, 'r') as f:

        labels_list = [line.translate(None, ''.join('\n')).split(': ')
                       for line in f if ':' in line]

    return labels_list

def find_video_labels_pairs(dir_imgs,
                           dir_segm_labels,
                           fpath_labels_list,
                           fpath_labels_list_subset,
                           dst_prefix,
                           dir_dst,
                           val_list=None):
    '''
    Fine intersection of filename in both directories and create
    one lmdb directory for each
    val_list - list of entities to exclude from train (validation subset e.g. ['2008_000002', '2010_000433'])
    '''
    


