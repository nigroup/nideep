'''
Created on Dec 10, 2015

@author: kashefy
'''
import os
import numpy as np
from scipy import io
import dataset_utils as du
import fileSystemUtils as fs

def get_labels_lut(labels_list, labels_subset):
    """
    Generate a look-up-table for mapping labels from a list to a subset
    Unmapped labels are mapped to class id zero.
    Can be used for selecting a subset of classes and grouping everything else.
    
    labels_list -- full list of (label name, class id) pairs
    labels_subset -- subset of pairs to keep
    """
    pairs = []
    len_labels_list = len(labels_list)
    max_src_idx = len(labels_list)-1
    for id_, name in labels_subset:
        
        found = False
        idx = 0
        while idx < len_labels_list and not found:
            
            id_src, name_src = labels_list[idx]
            
            if name_src == name:
                
                src_idx = int(id_src)
                pairs.append([src_idx, int(id_)])
                
                max_src_idx = max(max_src_idx, src_idx)
                
                found = True
            
            idx += 1
            
        if not found:
            print "Could not find %s" % name
    
    #print len(labels_list)
    sz = max(max_src_idx+1, len(labels_list)) + 1
    labels_lut = np.zeros((sz,), dtype='int')
    #print pairs
    for src, _ in pairs:
        labels_lut[src] = src
            
    return labels_lut

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

def read_matfiles(paths_src, fieldname, lut=None):
    '''
    Generate LMDB file from set of mat files with integer data
    Source: https://github.com/BVLC/caffe/issues/1698#issuecomment-70211045
    credit: Evan Shelhamer
    
    '''
    for path_ in paths_src:
        content_field = io.loadmat(path_)[fieldname]
        # get shape (1,H,W)
        while len(content_field.shape) < 3:
            content_field = np.expand_dims(content_field, axis=0)
        content_field = content_field.astype(int)
        
        import matplotlib.pyplot as plt
        import os
        print os.path.basename(path_dst), content_field.max()
        content_field[0,0,0]=0
        content_field[0,0,1]=454
        plt.figure()
        plt.imshow(content_field[0])
        from sets import Set
        set_1 = Set()
        [set_1.add(int(x)) for x in content_field.flatten()]
        
        if lut is not None:
            content_field = lut(content_field)
        
        set_2 = Set()
        [set_2.add(int(x)) for x in content_field.flatten()]
        
        print os.path.basename(path_dst), content_field.max(), len(set_1), len(set_2)
            
        plt.figure()
        plt.imshow(content_field[0])
        plt.show()

    return 0

def pascal_context_stats(dir_imgs,
                         dir_segm_labels,
                           fpath_labels_list,
                           fpath_labels_list_subset,
                           val_list=None):
    '''
    Fine intersection of filename in both directories and create
    one lmdb directory for each
    val_list - list of entities to exclude from train (validation subset e.g. ['2008_000002', '2010_000433'])
    '''        
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
                
        # ground truth
        paths_segm_labels_train = [paths_segm_labels[i] for i in train_idx]
        read_matfiles(paths_segm_labels_train, 'LabelMap',
                                 lut=apply_labels_lut)
        
        print 'val'
        paths_segm_labels_val = [paths_segm_labels[i] for i in val_idx]
        read_matfiles(paths_segm_labels_val, 'LabelMap',
                                 lut=apply_labels_lut)
        
        
        

if __name__ == '__main__':
    
    val_list_path = os.path.expanduser('~/data/PASCAL-Context/val_59.txt')
    with open(val_list_path, 'r') as f:
        val_list = f.readlines()
        val_list = [l.translate(None, ''.join('\n')) for l in val_list if len(l) > 0]
    
    pascal_context_stats(os.path.expanduser('~/data/VOCdevkit/VOC2012/JPEGImages'),
                           os.path.expanduser('~/data/PASCAL-Context/trainval'),
                           os.path.expanduser('~/data/PASCAL-Context/labels.txt'),
                           os.path.expanduser('~/data/PASCAL-Context/59_labels.txt'),
                           val_list=val_list
                           )
    
    print "size: %d" % nt, nv, fpath_imgs_train, fpath_labels_train, fpath_imgs_val, fpath_labels_val
    
    pass