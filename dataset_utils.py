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
    
    src -- dataset size (int) or full range of indices
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