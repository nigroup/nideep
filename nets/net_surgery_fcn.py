'''
Created on Jul 21, 2015

@author: kashefy
'''
import os
import numpy as np

CAFFE_ROOT = '/home/kashefy/src/caffe/'
if CAFFE_ROOT is not None:
    import sys
    sys.path.insert(0,  os.path.join(CAFFE_ROOT, 'python'))
import caffe

def make_fully_conv(path_model_src,
                    path_weights_src,
                    path_model_full_conv,
                    param_pairs,
                    path_weights_dst
                    ):
    
    # PART A
    # Load the original network and extract the fully connected layers' parameters.
    net = caffe.Net(path_model_src, path_weights_src, caffe.TEST)
    
    params = [src for src, _ in param_pairs]
    
    # fc_params = {name: (weights, biases)}
    fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}
    
    print "Original dimensions:"
    for fc in params:
        print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)
        
    # PART B
    # Load the fully convolutional network to transplant the parameters.
    net_full_conv = caffe.Net(path_model_full_conv, 
                              path_weights_src,
                              caffe.TEST)
    params_full_conv = [dst for _, dst in param_pairs]
    # conv_params = {name: (weights, biases)}
    conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}
    
    for conv in params_full_conv:
        print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)
        
    #Let's transplant!

    for pr, pr_conv in zip(params, params_full_conv):
        conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
        conv_params[pr_conv][1][...] = fc_params[pr][1]
        
    # save new weights
    net_full_conv.save(path_weights_dst)
    
    for pr, pr_conv in zip(params, params_full_conv):
        print pr_conv
         
        raw_input()
        print net_full_conv.params[pr_conv][0].data
    
    return 0
    import matplotlib.pyplot as plt
    
    # load input and configure preprocessing
    im = caffe.io.load_image(os.path.join(CAFFE_ROOT, 'examples/images/cat.jpg'))
    transformer = caffe.io.Transformer({'data': net_full_conv.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(os.path.join(CAFFE_ROOT, 'python/caffe/imagenet/ilsvrc_2012_mean.npy')).mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)
    # make classification map by forward and print prediction indices at each location
    out = net_full_conv.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
    print out['prob'][0].argmax(axis=0)
    # show net input and confidence map (probability of the top prediction at each location)
    plt.subplot(1, 2, 1)
    plt.imshow(transformer.deprocess('data', net_full_conv.blobs['data'].data[0]))
    plt.subplot(1, 2, 2)
    plt.imshow(out['prob'][0, 281], interpolation='nearest')
    plt.show()

def main(args):
    
    caffe.set_mode_cpu()
    
#     #param_pairs = [('fc6', 'fc6'), ('fc7', 'fc7'), ('fc8', 'fc8')]
#     param_pairs = [('fc6', 'fc6-conv'),
#                    ('fc7', 'fc7-conv'),
#                    ('fc8', 'fc8-conv')]
#     make_fully_conv(os.path.join(CAFFE_ROOT, 'models/bvlc_reference_caffenet/deploy.prototxt'),
#                     os.path.join(CAFFE_ROOT, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'),
#                     os.path.join(CAFFE_ROOT, 'examples/net_surgery/bvlc_caffenet_full_conv.prototxt'),
#                     param_pairs,
#                     os.path.join(CAFFE_ROOT, 'examples/net_surgery/bvlc_caffenet_full_conv.caffemodel'),
#                     )
    
    param_pairs = [('fc6', 'fc6-conv'),
                   ('fc7', 'fc7-conv'),
                   ('fc8', 'fc8-conv')]
    make_fully_conv('/home/kashefy/data/models/vgg-16/VGG_ILSVRC_16_layers_deploy.prototxt',
                    '/home/kashefy/data/models/vgg-16/VGG_ILSVRC_16_layers.caffemodel',
                    '/home/kashefy/data/models/vgg-16/VGG_ILSVRC_16_layers_fcn_deploy.prototxt',
                    param_pairs,
                    '/home/kashefy/data/models/vgg-16/VGG_ILSVRC_16_layers_fcn.caffemodel',
                    )
    
    
    return 0

if __name__ == '__main__':
    
    main(None)
    
    pass