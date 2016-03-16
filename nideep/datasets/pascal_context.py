'''
Created on Jul 21, 2015

@author: kashefy
'''
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
from nideep.iow.read_img import read_img_cv2, read_img_PIL

if __name__ == '__main__':
    
    caffe.set_mode_cpu()
    
    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    path_img = '/home/kashefy/data/VOCdevkit/VOC2012/JPEGImagesX/2008_000015.jpg'
    
    bgr_mean = np.array((104.00698793,116.66876762,122.67891434))
    im = Image.open(path_img)
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    print in_.shape
    print in_
    in_ -= bgr_mean
    print in_
    in_ = in_.transpose((2,0,1))
    
    in_ = read_img_PIL(path_img, mean=bgr_mean)
    
    print 'in_'
    print in_[0, 0, 0:6]
    print in_[1, 0, 0:6]
    print in_[2, 0, 0:6]
    
    in2 = read_img_cv2(path_img, mean=bgr_mean)
    print in2.shape
    #in2[0, :, :] -= 104.00698793
    #in2[1, :, :] -= 116.66876762
    #in2[2, :, :] -= 122.67891434
    
    print in2[0, 0, 0:6]
    print in2[1, 0, 0:6]
    print in2[2, 0, 0:6]
    
    print np.all(in_ == in2)
    print in_[in_ != in2]
    print in2[in_ != in2]
    return 0
        
    # load net
    path_model = '/home/kashefy/data/models/fcn_segm/fcn-32s-Pascal-context/deploy.prototxt'
    path_weights = '/home/kashefy/data/models/fcn_segm/fcn-32s-Pascal-context/fcn-32s-pascalcontext.caffemodel'
    net = caffe.Net(path_model, path_weights, caffe.TEST)
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_    
    

    
        
    # run net and take argmax for prediction
#     net.forward()
#     out = net.blobs['score'].data[0].argmax(axis=0)
#     
#     
#     print 'data after fwd'
#     print net.blobs['data'].data[net.blobs['data'].data.shape[0]/2-3:net.blobs['data'].data.shape[0]/2+3,
#               net.blobs['data'].data.shape[1]/2-3:net.blobs['data'].data.shape[1]/2+3]
#     
#     print 'out'
#     print out[out.shape[0]/2-3:out.shape[0]/2+3,
#               out.shape[1]/2-3:out.shape[1]/2+3]
#     plt.imshow(out)
#     plt.show()
    
    pass