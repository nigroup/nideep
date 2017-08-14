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

    @patch('nideep.datasets.mnist.mnist_tf.extract_images')
    def test_read(self, mock_extract):
        mock_extract.return_value = np.empty_like((100000, 28, 28))
        d = MNIST.read_data_sets(self.dir_tmp)
        assert_equals(d.train.num_examples, 55000)
        assert_equals(d.validation.num_examples, 5000)
        assert_equals(d.test.num_examples, 10000)
        
    @patch('nideep.datasets.mnist.mnist_tf.extract_images')
    def test_validation_size(self, mock_extract):
        mock_extract.return_value = np.empty_like((100000, 28, 28))
        for sz in range(5000, 55000, 5000):
            yield self.check_validation_size, sz

    def check_validation_size(self, sz):
        d = MNIST.read_data_sets(self.dir_tmp, validation_size=sz)
        assert_equals(d.train.num_examples, 60000-sz)
        assert_equals(d.validation.num_examples, sz)
        assert_equals(d.test.num_examples, 10000)

    def test_shuffle(self):
        d1 = MNIST.read_data_sets(self.dir_tmp)
        d2 = MNIST.read_data_sets(self.dir_tmp)
        assert_false(np.array_equal(d1.train.images, d2.train.images))
        assert_false(np.array_equal(d1.train.labels, d2.train.labels))
        assert_false(np.array_equal(d1.validation.images, d2.validation.images))
        assert_false(np.array_equal(d1.validation.labels, d2.validation.labels))
        assert_true(np.array_equal(d1.test.images, d2.test.images))
        assert_true(np.array_equal(d1.test.labels, d2.test.labels))
#        
        