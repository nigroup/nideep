from __future__ import division
import os

CAFFE_ROOT = '/home/kashefy/src/caffe/'
BASE_WEIGHTS='/home/kashefy/data/models/vgg-16/VGG_ILSVRC_16_layers_fcn.caffemodel'
PATH_SOLVER='/home/kashefy/data/models/fcn_segm/fcn-32s-Pascal-context/t2/solver2.prototxt'

if CAFFE_ROOT is not None:
    import sys
    sys.path.insert(0,  os.path.join(CAFFE_ROOT, 'python'))
import caffe
import numpy as np

# make a bilinear interpolation kernel
# credit @longjon
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print 'input + output channels need to be the same'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt

# base net -- follow the editing model parameters example to make
# a fully convolutional VGG16 net.
# http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/net_surgery.ipynb

# init
caffe.set_mode_cpu()

solver = caffe.SGDSolver(PATH_SOLVER)

# do net surgery to set the deconvolution weights for bilinear interpolation
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
interp_surgery(solver.net, interp_layers)

# copy base weights for fine-tuning
solver.net.copy_from(BASE_WEIGHTS)



