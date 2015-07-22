'''
Created on Jul 21, 2015

@author: kashefy
'''
import os
import numpy as np
import fileSystemUtils as fs

import cv2 as cv2
import cv2.cv as cv

import os
CAFFE_ROOT = '/home/kashefy/src/caffe_forks/bvlc/'
os.chdir(CAFFE_ROOT)
import sys
sys.path.insert(0, './python')
import caffe

def main(args):
    
    caffe.set_mode_cpu()
    solver = caffe.SGDSolver('/media/win/Users/woodstock/dev/data/models/fcn_segm/solver2.prototxt')
    
    niter = 20
    
    # the main solver loop
    for it in range(niter):
        solver.step(1)  # SGD by Caffe
        
    return 0

if __name__ == '__main__':
    
    main(None)
    
    pass