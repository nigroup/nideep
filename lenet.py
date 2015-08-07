'''
Created on Jul 14, 2015

@author: kashefy
'''
import os
import sys
import caffe
import cv2 as cv2
import cv2.cv as cv

from pylab import *

from caffe import layers as L
from caffe import params as P

def lenet(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.ip1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    return n.to_proto()
    
if __name__ == '__main__':
    
    os.chdir('/home/kashefy/src/caffe/')
    
    with open('examples/mnist/lenet_auto_train.prototxt', 'w') as f:
        f.write(str(lenet('examples/mnist/mnist_train_lmdb', 64)))
        
    with open('examples/mnist/lenet_auto_test.prototxt', 'w') as f:
        f.write(str(lenet('examples/mnist/mnist_test_lmdb', 100)))
        
        
    #caffe.set_device(0) # for gpu mode
    caffe.set_mode_cpu()
    solver = caffe.SGDSolver('examples/mnist/lenet_auto_solver.prototxt')
    
#     solver.net.forward()  # train net
#     solver.test_nets[0].forward()  # test net (there can be more than one)
# 
#     imshow(solver.test_nets[0].blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray')
#     print solver.test_nets[0].blobs['label'].data[:8]

#     niter = 10
#     for it in xrange(niter):
#         print it
#         solver.step(500)
#         print "step"
#     
#         imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(4, 5, 5, 5)
#            .transpose(0, 2, 1, 3).reshape(4*5, 5*5), cmap='gray')
#         show()

    niter = 200
    test_interval = 25
    # losses will also be stored in the log
    train_loss = zeros(niter)
    test_acc = zeros(int(np.ceil(niter / test_interval)))
    output = zeros((niter, 8, 10))
    
    filt_hist = []
    
    # the main solver loop
    for it in range(niter):
        solver.step(1)  # SGD by Caffe
        print solver.test_nets[0].params['conv1'][0].data[1,0,:,:]
        
        # store the train loss
        train_loss[it] = solver.net.blobs['loss'].data
        
        # store the output on the first test batch
        # (start the forward pass at conv1 to avoid loading new data)
        solver.test_nets[0].forward(start='conv1')
        output[it] = solver.test_nets[0].blobs['ip2'].data[:8]
        
        # run a full test every so often
        # (Caffe can also do this for us and write to a log, but we show here
        #  how to do it directly in Python, where more complicated things are easier.)
        if it % test_interval == 0:
            print 'Iteration', it, 'testing...'
            correct = 0
            for test_it in range(100):
                solver.test_nets[0].forward()
                correct += sum(solver.test_nets[0].blobs['ip2'].data.argmax(1)
                               == solver.test_nets[0].blobs['label'].data)
            test_acc[it // test_interval] = correct / 1e4
            
            filt_hist.append(solver.test_nets[0].params['conv1'][0].data)
            

    fig = figure()
    weights = filt_hist[-1]
    n = int(np.ceil(np.sqrt(weights.shape[0])))
    for i, f in enumerate(weights):
        ax = fig.add_subplot(n, n, i+1)
        ax.axis('off')
        cm = None
        if f.ndim > 2 and f.shape[0]==1:
            f = f.reshape(f.shape[1:])
        if f.ndim == 2 or f.shape[0]==1:
            cm = 'gray'
        imshow(f, cmap=cm)
    show()
        
    _, ax1 = subplots()
    ax2 = ax1.twinx()
    ax1.plot(arange(niter), train_loss)
    ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss')
    ax2.set_ylabel('test accuracy')
    
    show()
    
    n = int(np.ceil(np.sqrt(d.shape[0])))
    fig = figure()
    for i, f in enumerate(d):
        ax = fig.add_subplot(n, n, i+1)
        ax.axis('off')
        print f.shape
        cm = None
        if f.ndim > 2 and f.shape[0]==1:
            f = f.reshape(f.shape[1:])
        if f.ndim == 2 or f.shape[0]==1:
            cm = 'gray'
        imshow(f, cmap=cm)
            
    
    pass