import os

def gen_paths(dir_src, func_filter=None):
    
    if func_filter is None:
        def func_filter(fileName):
            return True
        
    dst = []
    
    for root_, dir_, files_ in os.walk(dir_src):
        
        for fname in files_:
            
            p = os.path.join(root_, fname)
            if func_filter(p):
                dst.append(p)
        #print root_, dir_, files_       
    
    return dst

def filter_is_img(fname):
    
    _, ext = os.path.splitext(fname)
    
    return ext in ['.bmp', '.jpg', '.png', '.tif']

def fname_pairs(paths_a, paths_b):
    
    paths_dst = []
    len_b = len(paths_b)
    cursor_b = 0
    
    for pa in paths_a:
        
        base_a, _ = os.path.splitext(os.path.basename(pa))
        
        found_b = False
        
        cursor_b_end = cursor_b-1
        if cursor_b_end < 0:
            cursor_b_end = len_b-1
            
        while cursor_b != cursor_b_end and not found_b:
            
            pb = paths_b[cursor_b]
            base_b, _ = os.path.splitext(os.path.basename(pb))
            if base_b.startswith(base_a):
                
                paths_dst.append([pa, pb])
                found_b = True

            cursor_b += 1
            if cursor_b >= len_b:
                cursor_b = 0
    
    return paths_dst
