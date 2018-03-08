'''
Created on Jul 23, 2015

@author: kashefy
'''
import numpy as np
from PIL import Image
import cv2 as cv2
import caffe
from nideep.blobs import mat_utils as mu

def read_img_PIL(fpath, mean=None):
    '''
    load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    '''
    img = Image.open(fpath)  # pixel value range per channel: [0, 255]

    img_dat = np.array(img, dtype=np.float32)

    # RGB to BGR
    img_dat = img_dat[:, :, ::-1]

    # per-channel mean subtraction
    if mean is not None:
        img_dat -= mean

    # reorder dimensions
    img_dat = mu.hwc_to_chw(img_dat)

    return img_dat

def read_img_cv2(fpath, mean=None):
    '''
    load image in BGR, subtract mean, and make dims C x H x W for Caffe
    '''
    img_dat = cv2.imread(fpath)  # pixel value range per channel: [0, 255]

    # channels already in BGR order

    # per-channel mean subtraction
    if mean is not None:
        img_dat = img_dat.astype(np.float32)
        img_dat -= mean
        img_dat = img_dat.astype(np.float)

    # reorder dimensions
    img_dat = mu.hwc_to_chw(img_dat)

    # casting to np.float enables plugging into protobuf

    return img_dat

def read_img_caf(fpath, mean=None):
    '''
    load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    '''
    img_dat = caffe.io.load_image(fpath)  # pixel value range per channel: [0, 1]

    img_dat *= 255.

    # RGB to BGR
    img_dat = img_dat[:, :, ::-1]

    # per-channel mean subtraction
    if mean is not None:
        img_dat -= mean

    # reorder dimensions
    img_dat = mu.hwc_to_chw(img_dat)

    return img_dat

if __name__ == '__main__':

    fpath = '/media/win/Users/woodstock/dev/data/VOCdevkit/VOC2012/JPEGImages/2008_000064.jpg'
    img_pil = read_img_PIL(fpath, mean=np.array((104.00698793, 116.66876762, 122.67891434)))
    img_cv2 = read_img_cv2(fpath, mean=np.array((104.00698793, 116.66876762, 122.67891434)))
    img_caf = read_img_caf(fpath, mean=np.array((104.00698793, 116.66876762, 122.67891434)))
    print(img_pil.shape, img_cv2.shape, img_caf.shape)
    print(mg_pil[:, 0:2, 0:2])
    print(img_cv2[:, 0:2, 0:2])
    print(img_caf[:, 0:2, 0:2])

    pass
