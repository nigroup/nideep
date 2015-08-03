'''
Created on Jul 21, 2015

@author: kashefy
'''
import os
import numpy as np
import fileSystemUtils as fs

import cv2 as cv2
import cv2.cv as cv

CAFFE_ROOT = '/home/kashefy/src/caffe/'
if CAFFE_ROOT is not None:
    import sys
    sys.path.insert(0,  os.path.join(CAFFE_ROOT, 'python'))
import caffe

def main(args):
    
    caffe.set_mode_cpu()
    
    

if __name__ == '__main__':
    
    main(None)
    
    pass