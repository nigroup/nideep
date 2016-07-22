'''
Created on Jul 21, 2015

@author: kashefy
'''
import os
import caffe

def make_fully_conv(path_model_src,
                    path_weights_src,
                    path_model_full_conv,
                    param_pairs,
                    path_weights_dst
                    ):

    # PART A: Load the original network and extract the fully connected layers' parameters.
    net_full_cnnct = caffe.Net(path_model_src, path_weights_src, caffe.TEST)

    params = [src for src, _ in param_pairs]

    # fc_params = {name: (weights, biases)}
    fc_params = {pr: (net_full_cnnct.params[pr][0].data, net_full_cnnct.params[pr][1].data) for pr in params}

    print "Original dimensions:"
    for fc in params:
        print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

    # PART B: Load the fully convolutional network to transplant the parameters.
    net_full_conv = caffe.Net(path_model_full_conv, path_weights_src, caffe.TEST)
    params_full_conv = [dst for _, dst in param_pairs]

    # conv_params = {name: (weights, biases)}
    conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

    for conv in params_full_conv:
        print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)

    # Let's transplant!
    for pr, pr_conv in zip(params, params_full_conv):
        conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
        conv_params[pr_conv][1][...] = fc_params[pr][1]

    # save new weights
    net_full_conv.save(path_weights_dst)

    for pr, pr_conv in zip(params, params_full_conv):
        print pr_conv
        print net_full_conv.params[pr_conv][0].data

    return 0

if __name__ == '__main__':

    caffe.set_mode_cpu()

    param_pairs = [('fc6', 'fc6-conv'),
                   ('fc7', 'fc7-conv'),
                   ('fc8', 'fc8-conv')]

    make_fully_conv(os.path.expanduser('~/models/vgg-16/VGG_ILSVRC_16_layers_deploy.prototxt'),
                    os.path.expanduser('~/models/vgg-16/VGG_ILSVRC_16_layers.caffemodel'),
                    os.path.expanduser('~/models/vgg-16/VGG_ILSVRC_16_layers_fcn_deploy_151208.prototxt'),
                    param_pairs,
                    os.path.expanduser('~/models/vgg-16/VGG_ILSVRC_16_layers_fcn_151208.caffemodel'),
                    )
