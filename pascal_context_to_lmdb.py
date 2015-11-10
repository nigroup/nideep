'''
Created on Jul 21, 2015

@author: kashefy
'''
import os
import numpy as np
import to_lmdb
import dataset_utils as du
import fileSystemUtils as fs

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

def pascal_context_to_lmdb(dir_imgs,
                           dir_segm_labels,
                           fpath_labels_list,
                           fpath_labels_list_subset,
                           dst_prefix,
                           dir_dst,
                           CAFFE_ROOT=None,
                           val_list=None):
    '''
    Fine intersection of filename in both directories and create
    one lmdb directory for each
    val_list - list of entities to exclude from train (validation subset e.g. ['2008_000002', '2010_000433'])
    '''
    if dst_prefix is None:
        dst_prefix = ''
        
    labels_list = get_labels_list(fpath_labels_list)
    labels_59_list = get_labels_list(fpath_labels_list_subset)
    
    #print labels_list
    #print labels_59_list
    labels_lut = du.get_labels_lut(labels_list, labels_59_list)
    def apply_labels_lut(m):
        return labels_lut[m]
    
    paths_imgs = fs.gen_paths(dir_imgs, fs.filter_is_img)
    
    paths_segm_labels = fs.gen_paths(dir_segm_labels)
     
    paths_pairs = fs.fname_pairs(paths_imgs, paths_segm_labels)    
    paths_imgs, paths_segm_labels = map(list, zip(*paths_pairs))
    
    #for a, b in paths_pairs:
    #    print a,b
    
    if val_list is not None:
        # do train/val split
        
        train_idx, val_idx = du.get_train_val_split_from_names(paths_imgs, val_list)
        
        # images
        paths_imgs_train = [paths_imgs[i] for i in train_idx]
        fpath_lmdb_imgs_train = os.path.join(dir_dst,
                                       '%scontext_imgs_train_lmdb' % dst_prefix)
        to_lmdb.imgs_to_lmdb(paths_imgs_train,
                             fpath_lmdb_imgs_train)
        
        paths_imgs_val = [paths_imgs[i] for i in val_idx]
        fpath_lmdb_imgs_val = os.path.join(dir_dst,
                                       '%scontext_imgs_val_lmdb' % dst_prefix)
        to_lmdb.imgs_to_lmdb(paths_imgs_val,
                             fpath_lmdb_imgs_val)
        
        # ground truth
        paths_segm_labels_train = [paths_segm_labels[i] for i in train_idx]
        fpath_lmdb_segm_labels_train = os.path.join(dir_dst,
                                              '%scontext_labels_train_lmdb' % dst_prefix)
        to_lmdb.matfiles_to_lmdb(paths_segm_labels_train,
                                 fpath_lmdb_segm_labels_train,
                                 'LabelMap',
                                 lut=apply_labels_lut)
        
        paths_segm_labels_val = [paths_segm_labels[i] for i in val_idx]
        fpath_lmdb_segm_labels_val = os.path.join(dir_dst,
                                              '%scontext_labels_val_lmdb' % dst_prefix)
        to_lmdb.matfiles_to_lmdb(paths_segm_labels_val,
                                 fpath_lmdb_segm_labels_val,
                                 'LabelMap',
                                 lut=apply_labels_lut)
        
        return len(paths_imgs_train), len(paths_imgs_val),\
            fpath_lmdb_imgs_train, fpath_lmdb_segm_labels_train, fpath_lmdb_imgs_val, fpath_lmdb_segm_labels_val
        
    else:
        fpath_lmdb_imgs = os.path.join(dir_dst,
                                       '%scontext_imgs_lmdb' % dst_prefix)
        to_lmdb.imgs_to_lmdb(paths_imgs,
                             fpath_lmdb_imgs)
        
        fpath_lmdb_segm_labels = os.path.join(dir_dst,
                                              '%scontext_labels_lmdb' % dst_prefix)
        to_lmdb.matfiles_to_lmdb(paths_segm_labels,
                                 fpath_lmdb_segm_labels,
                                 'LabelMap',
                                 lut=apply_labels_lut)
        
        return len(paths_imgs), fpath_lmdb_imgs, fpath_lmdb_segm_labels

if __name__ == '__main__':
    
    val_list_path = '/home/kashefy/data/PASCAL-Context/val_59.txt'
    with open(val_list_path, 'r') as f:
        val_list = f.readlines()
        val_list = [l.translate(None, ''.join('\n')) for l in val_list if len(l) > 0]
    
    nt, nv, fpath_imgs_train, fpath_labels_train, fpath_imgs_val, fpath_labels_val = \
    pascal_context_to_lmdb(os.path.expanduser('~/data/VOCdevkit/VOC2012/JPEGImagesX'),
                           os.path.expanduser('~/data/PASCAL-Context/trainval'),
                           os.path.expanduser('~/data/PASCAL-Context/labels.txt'),
                           os.path.expanduser('~/data/PASCAL-Context/59_labels.txt'),
                           '',
                           os.path.expanduser('~/data/PASCAL-Context/'),
                           val_list=val_list
                           )
    
    print "size: %d" % nt, nv, fpath_imgs_train, fpath_labels_train, fpath_imgs_val, fpath_labels_val
    
    pass