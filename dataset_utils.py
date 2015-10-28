'''
Created on Oct 28, 2015

@author: kashefy
'''

def get_train_val_split(src, val_list):
    """
    Get indices split for train and validation subsets
    
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