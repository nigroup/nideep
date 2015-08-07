'''
Created on Jul 18, 2015

@author: kashefy
'''
import os
import cv2 as cv2
import cv2.cv as cv
import to_lmdb

CAFFE_ROOT = '/home/kashefy/src/caffe_forks/bvlc/'
    
if CAFFE_ROOT is not None:
    os.chdir(CAFFE_ROOT)
    import sys
    sys.path.insert(0, './python')
import caffe

def main(args):
    
    paths_in = ['/media/win/Users/woodstock/dev/data/lena.png']
    
    caffe.set_mode_cpu()
    
    path_lmdb_train = 'image-lmdb'
    path_lmdb_test = path_lmdb_train
    
    to_lmdb.imgs_to_lmdb(paths_in, path_lmdb_train)
    
    with open('train.prototxt', 'w') as f:
        f.write(str(to_lmdb.gen_net(path_lmdb_train, 1)))
    
    with open('test.prototxt', 'w') as f:
        f.write(str(to_lmdb.gen_net(path_lmdb_test, 1)))
    
    solver = caffe.SGDSolver('auto_solver.prototxt')
    
    for i in xrange(len(paths_in)):
        
        solver.net.forward()  # train net
        print solver.net.blobs['data'].data.shape
        
        d = solver.net.blobs['data'].data
        sh = d.shape
        d = d.reshape(sh[1], sh[2], sh[3])
        y = cv2.merge([d[0, :, :], d[1, :, :], d[2, :, :]])
        print y.dtype
        print y
        cv2.imshow('y', y)
        cv2.waitKey()
    
    return 0

if __name__ == '__main__':
    
    main(None)
    
    pass