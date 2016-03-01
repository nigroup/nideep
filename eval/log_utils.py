'''
Created on Oct 1, 2015

@author: kashefy
'''
import os
from itertools import product

def get_rel_readme_path(p):
    """
    Get README file in same directory
    """
    if os.path.isfile(p):
        p = os.path.dirname(p)
        
    for fname, ext in product(["README", "ReadMe", "Readme", "readme"],
                              ['', '.txt', '.md']):
        
        fpath = os.path.join(p, fname+ext)
        if os.path.isfile(fpath):
            return fpath

def readme_to_str(fpath):
    
    with open (fpath, "r") as f:
        return f.read()

def is_caffe_log(p):
    
    if os.path.isfile(p):
        
        b = os.path.basename(p)
        if b.startswith('caffe'):
            
            with open(p, 'r') as f:
                
                line = f.readline()
                return 'log file' in line.strip().lower()
        else:
            return False
        
    else:
        return False
    
def is_caffe_info_log(p):
    
    return '.INFO.' in os.path.basename(p) and is_caffe_log(p)

def find_line(p, substr):
    
    found = False
    with open(p, 'r') as f:
    
        for line in f:
            found = substr in line
            if found:
                return line
            
    return None

def pid_from_str(s):
    
    id_ = -1
    try: 
        id_ = int(s)
        
    except ValueError:
        pass

    return id_

def pid_from_logname(p):
    
    _, ext = os.path.splitext(p)
    ext = ext.replace(".", "")
    
    return pid_from_str(ext)

def read_pid(p):
    
    id_ = pid_from_logname(p)
    if id_ < 0:
        
        l = find_line(p, "Starting Optimization")
        
        if l is None:
            raise IOError("Failed to parse log for pid. (%s)" % p)
        
        id_ = pid_from_str(l.split(' ')[2])
        
    return id_

def is_complete(p):
    """
    Determine if the logged optimization was completed
    """
    l = find_line(p, "Optimization Done")
    return l is not None

            
    