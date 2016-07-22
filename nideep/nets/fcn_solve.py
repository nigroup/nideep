import os
import numpy as np
import caffe

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


def init_up_bilinear(net, path_base_weights, key='up'):
    """
    base net -- follow the editing model parameters example to make a fully convolutional VGG16 net.
    http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/net_surgery.ipynb
    """
    
    # do net surgery to set the deconvolution weights for bilinear interpolation
    interp_layers = [k for k in net.params.keys() if key in k]
    interp_surgery(net, interp_layers)
    
    # copy base weights for fine-tuning
    net.copy_from(path_base_weights)
    
    return

if __name__ == '__main__':
    
    caffe.set_mode_cpu()
    
    solver = caffe.SGDSolver(os.path.expanduser('~/models/fcn_segm/fcn-32s-Pascal-context/tx3/solver.prototxt'))
    init_up_bilinear(solver.net, os.path.expanduser('~/models/vgg-16/VGG_ILSVRC_16_layers_fcn.caffemodel'))
    solver.net.save(os.path.expanduser('~/models/fcn_segm/fcn-32s-Pascal-context/tx3/fcn.caffemodel'))
