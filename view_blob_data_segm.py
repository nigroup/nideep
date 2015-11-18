'''
Created on Nov 10, 2015

@author: kashefy
'''
import numpy as np
import cv2 as cv2
import cv2.cv as cv
import caffe

def view_blob_label_segm(nb_imgs, path_solver):
    
    solver = caffe.SGDSolver(path_solver)
     
    for _ in xrange(nb_imgs):
         
        solver.net.forward()  # train net
         
        d = solver.net.blobs['data'].data
        print d.shape
        d = np.squeeze(d, axis=(0,)) # get rid of batch elements dimensions
        y = cv2.cvtColor(cv2.merge([ch for ch in d]), cv.CV_RGB2BGR)
         
        #print y.dtype, y.max()
         
        cv2.imshow('data', y)
         
        d = solver.net.blobs['label'].data
        print d.shape
        d = np.squeeze(d, axis=(0,))
        
        print d
        
        cv2.waitKey()
        
    return 0


if __name__ == '__main__':
    
    view_blob_label_segm(2, os.path.expanduser('~/models/fcn_segm/solver2.prototxt')
    
    pass