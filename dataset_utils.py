'''
Created on Oct 28, 2015

@author: kashefy
'''

def get_train_val_split_from_names(src, val_list):
    """
    Get indices split for train and validation entity names subset
    
    src -- list of all entities in dataset
    val_list -- contains entities that belong to the validation subset
    
    """    
    train_idx = []
    val_idx = []
    
    len_ = len(val_list)
    
    for i, x in enumerate(src):
        
        found = False
        j = 0
        while j < len_ and not found:
            
            found = val_list[j] in x            
            j += 1
            
        if found:
            val_idx.append(i)
        else:
            train_idx.append(i)
    
    return train_idx, val_idx

def get_train_val_split_from_idx(src, val_list):
    """
    Get indices split for train and validation subsets
    
    src -- dataset size (int) or full range of indices (list)
    val_list -- indices that belong to the validation subset
    
    """    
    train_idx = []
    val_idx = val_list
    
    if not hasattr(src, '__iter__'):
        src = range(src)
    
    for x in src:
        
        if x not in val_idx:
            train_idx.append(x)
    
    return train_idx, val_idx

def get_labels_lut(labels_list, labels_subset):
    """
    Generate a look-up-table for mapping labels from a list to a subset
    Unmapped labels are mapped to class id zero.
    Can be used for selecting a subset of classes and grouping everything else.
    
    labels_list -- full list of labels/class names
    labels_subset -- contains entities that belong to the validation subset
    """
    pairs = []
    len_labels_list = len(labels_list)
    for id_, name in labels_subset:
        
        found = False
        idx = 0
        while idx < len_labels_list and not found:
            
            id2, name2 = labels_list[idx]
            
            if name2 == name:
                pairs.append([int(id2), int(id_)])
                found = True
            
            idx += 1
            
        if not found:
            print "Could not find %s" % name
    
    #print len(labels_list)
    labels_lut = np.zeros((len(labels_list)+1,), dtype='int')
    #print pairs
    for src, dst in pairs:
        labels_lut[src] = dst
            
    return labels_lut