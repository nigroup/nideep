'''
Created on Aug 11, 2017

@author: kashefy
'''
import os
import tempfile
import shutil
import numpy as np
from nose.tools import assert_equals, assert_true, assert_false, \
    assert_list_equal
from mock import patch
from mnist_tf import MNIST

class TestMNISTTF:
    @classmethod
    def setup_class(self):
        self.dir_tmp = tempfile.mkdtemp()
        
    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.dir_tmp)
        
    def test_to_tf_record(self):
        import tensorflow as tf
        x = tf.placeholder(tf.float32, [None, 784])
        name_w = 'W'
        with tf.variable_scope("var_scope", reuse=None):
            W = tf.get_variable(name_w, shape=[784, 10],
                                initializer=tf.random_normal_initializer(stddev=0.1))
            b = tf.get_variable('b', shape=[10],
                                initializer=tf.constant_initializer(0.1))
        logits = tf.matmul(x, W) + b
        y = tf.nn.softmax(logits)
        y_ = tf.placeholder(tf.float32, [None, 10])
        cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(\
                labels=y_, logits=logits))
        train_step = tf.train.GradientDescentOptimizer(0.9).minimize(cross_entropy)
        init_op = tf.global_variables_initializer()
        one_hot = True
        shuffle = True
        orientations = range(-60,75,15)
        print(orientations)
#        mnist = MNIST.read_data_sets('MNIST_data', one_hot=True)
        fpath_list = MNIST.to_tf_record(os.path.join('MNIST_data', 'train.tfrecords'),
                           'MNIST_data',
                           one_hot=one_hot,
                           orientations=orientations)
        print fpath_list
        return

        img, label, label_orient = MNIST.read_and_decode_ops(\
                                                os.path.join('MNIST_data', 'train.tfrecords'),
                                               one_hot=one_hot,
                                               num_orientations=len(orientations))
#        import tensorflow as tf
        if shuffle:
            batch_xs_op, batch_ys_op, batch_os_op = tf.train.shuffle_batch([img, label, label_orient],
                                                    batch_size=64,
                                                    capacity=2000,
                                                    min_after_dequeue=1000,
                                                    num_threads=8
                                                    )
        else:
            batch_xs_op, batch_ys_op, batch_os_op = tf.train.batch([img, label, label_orient],
                                                    batch_size=64,
                                                    capacity=2000,
    #                                                min_after_dequeue=1000,
                                                    num_threads=8
                                                    )
        
        import matplotlib.pyplot as plt
        f, a = plt.subplots(3, 12, figsize=(10, 2))
        with tf.Session() as sess:
                    
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for i in range(3):
                val, l, orient = sess.run([batch_xs_op, batch_ys_op, batch_os_op])
                print(val.shape, l)
                for j in range(12):
                    a[i][j].imshow(np.reshape(val[j], (28, 28)), clim=(0.0, 1.0))
                    if one_hot:
                        a[i][j].set_title('%d at %.2f deg' % (np.argmax(l[j]), orientations[np.argmax(orient[j])]))
                    else:
                        a[i][j].set_title('%d at %.2f deg' % (l[j], orientations[orient[j]]))

            plt.show()
            return
            sess.run(init_op)
            for itr in range(1000):
                batch_xs, batch_ys = sess.run([batch_xs_op, batch_ys_op])
                _, c = sess.run([train_step, cross_entropy],
                     feed_dict={x: batch_xs, y_: batch_ys})
#                print(c)
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(sess.run(accuracy,
                           feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
            print(sess.run(accuracy,
                           feed_dict={x: mnist.validation.images, y_: mnist.validation.labels}))
            
            coord.request_stop()
            coord.join(threads)

#    def test_orient(self):
#        d1 = MNIST.read_data_sets_orient(self.dir_tmp, seed=123, orientations=range(-30,60,30))
            
#    @patch('nideep.datasets.mnist.mnist_tf.extract_images')
#    def test_read(self, mock_extract):
#        mock_extract.return_value = np.empty_like((100000, 28, 28))
#        d = MNIST.read_data_sets(self.dir_tmp)
#        assert_equals(d.train.num_examples, 55000)
#        assert_equals(d.validation.num_examples, 5000)
#        assert_equals(d.test.num_examples, 10000)
#        
#    @patch('nideep.datasets.mnist.mnist_tf.extract_images')
#    def test_validation_size(self, mock_extract):
#        mock_extract.return_value = np.empty_like((100000, 28, 28))
#        for sz in range(5000, 55000, 5000):
#            yield self.check_validation_size, sz
#
#    def check_validation_size(self, sz):
#        d = MNIST.read_data_sets(self.dir_tmp, validation_size=sz)
#        assert_equals(d.train.num_examples, 60000-sz)
#        assert_equals(d.validation.num_examples, sz)
#        assert_equals(d.test.num_examples, 10000)
#
#    def test_shuffle(self):
#        d1 = MNIST.read_data_sets(self.dir_tmp)
#        d2 = MNIST.read_data_sets(self.dir_tmp)
#        assert_false(np.array_equal(d1.train.images, d2.train.images))
#        assert_false(np.array_equal(d1.train.labels, d2.train.labels))
#        assert_false(np.array_equal(d1.validation.images, d2.validation.images))
#        assert_false(np.array_equal(d1.validation.labels, d2.validation.labels))
#        assert_true(np.array_equal(d1.test.images, d2.test.images))
#        assert_true(np.array_equal(d1.test.labels, d2.test.labels))
#
#    def test_seed(self):
#        d1 = MNIST.read_data_sets(self.dir_tmp, seed=123)
#        d2 = MNIST.read_data_sets(self.dir_tmp, seed=999)
#        d3 = MNIST.read_data_sets(self.dir_tmp, seed=123)
#        assert_false(np.array_equal(d1.train.images, d2.train.images))
#        assert_false(np.array_equal(d1.train.labels, d2.train.labels))
#        assert_false(np.array_equal(d1.validation.images, d2.validation.images))
#        assert_false(np.array_equal(d1.validation.labels, d2.validation.labels))
#        assert_true(np.array_equal(d1.test.images, d2.test.images))
#        assert_true(np.array_equal(d1.test.labels, d2.test.labels))
#        # same seed
#        assert_true(np.array_equal(d1.train.images, d3.train.images))
#        assert_true(np.array_equal(d1.train.labels, d3.train.labels))
#        assert_true(np.array_equal(d1.validation.images, d3.validation.images))
#        assert_true(np.array_equal(d1.validation.labels, d3.validation.labels))
#        assert_true(np.array_equal(d1.test.images, d3.test.images))
#        assert_true(np.array_equal(d1.test.labels, d3.test.labels))
#        
#    def test_data_scale(self):
#        from tensorflow.examples.tutorials.mnist import input_data
#        d0 = input_data.read_data_sets(self.dir_tmp, one_hot=True)
#        d1 = MNIST.read_data_sets(self.dir_tmp, one_hot=True)
#        assert_equals(d0.train.images.min(), 0)
#        assert_equals(d0.train.images.max(), 1)
#        assert_equals(d0.train.images.min(), d1.train.images.min())
#        assert_equals(d0.train.images.max(), d1.train.images.max())
#        assert_equals(d0.validation.images.min(), 0)
#        assert_equals(d0.validation.images.max(), 1)
#        assert_equals(d0.validation.images.min(), d1.validation.images.min())
#        assert_equals(d0.validation.images.max(), d1.validation.images.max())
#        assert_equals(d0.test.images.min(), 0)
#        assert_equals(d0.test.images.max(), 1)
#        assert_equals(d0.test.images.min(), d1.test.images.min())
#        assert_equals(d0.test.images.max(), d1.test.images.max())
        