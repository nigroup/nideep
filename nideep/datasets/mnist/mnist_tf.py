'''
Created on Jul 31, 2017

@author: kashefy
'''
import numpy as np
from tensorflow.python.framework import dtypes, random_seed
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet, read_data_sets
import tensorflow as tf
# imported for mocking
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class MNIST(object):
    '''
    classdocs
    '''
    @classmethod
    def read_data_sets(cls,
                       train_dir,
                       fake_data=False,
                       one_hot=False,
                       dtype=dtypes.float32,
                       reshape=True,
                       validation_size=5000,
                       seed=None):
        
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        np.random.seed(seed1 if seed is None else seed2)
    
        ds = read_data_sets(\
                                  train_dir,
                                  fake_data=fake_data,
                                  one_hot=one_hot,
                                  dtype=dtype,
                                  reshape=reshape,
                                  validation_size=0,
                                  seed=seed)
        perm0 = np.arange(ds.train.num_examples)
        np.random.shuffle(perm0)
        train_idxs = perm0[validation_size:]
        val_idxs = perm0[:validation_size]
        train = DataSet(\
                    np.multiply(ds.train.images[train_idxs], [1., 255.][dtype == dtypes.float32]), # will rescale to [0,1] inside
                    ds.train.labels[train_idxs],
                    fake_data=fake_data,
                    one_hot=one_hot,
                    dtype=dtype,
                    reshape=False, # already reshaped
                    seed=seed)
        validation = DataSet(\
                        np.multiply(ds.train.images[val_idxs], [1., 255.][dtype == dtypes.float32]), # will rescale to [0,1] inside
                        ds.train.labels[val_idxs],
                        fake_data=fake_data,
                        one_hot=one_hot,
                        dtype=dtype,
                        reshape=False, # already reshaped
                        seed=seed)
        return Datasets(train=train, validation=validation, test=ds.test)
    
    @classmethod
    def to_tf_record(cls, fpath, one_hot=False):
        mnist = MNIST.read_data_sets("MNIST_data", one_hot=one_hot, validation_size=5000)
        with tf.python_io.TFRecordWriter(fpath) as writer:
            for img, label in zip(mnist.train.images, mnist.train.labels):
                img_raw = img.tobytes()
                ## Convert the bytes back to image as follow:
                # image = Image.frombytes('RGB', (224,224), img_raw)
                # tl.visualize.frame(I=image, second=1, saveable=False, name='frame', fig_idx=1236)
                ## Write the data into TF format
                # image     : Feature + BytesList
                # label     : Feature + Int64List or FloatList
                # sentence  : FeatureList + Int64List , see Google's im2txt example
                if one_hot:
                    label_raw = label.astype(np.float32).tobytes()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        "label"     : _bytes_feature(label_raw),
                        'img_raw'   : _bytes_feature(img_raw),
                        }))
                else:
                    example = tf.train.Example(features=tf.train.Features(feature={
                        "label"     : _int64_feature(int(label)),
                        'img_raw'   : _bytes_feature(img_raw),
                        }))
                writer.write(example.SerializeToString())  # Serialize To String
        
    @classmethod        
    def read_and_decode_ops(cls, fpath, one_hot=False):
        # generate a queue with a given file name
        filename_queue = tf.train.string_input_producer([fpath])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)     # return the file and the name of file
        if one_hot:
            features = tf.parse_single_example(serialized_example,  # see parse_single_sequence_example for sequence example
                                               features={
                                                   'label': tf.FixedLenFeature([], tf.string),
                                                   'img_raw' : tf.FixedLenFeature([], tf.string),
                                               })
            label = tf.decode_raw(features['label'], tf.float32)            
#            label = tf.reshape(label, [10])           
            label.set_shape((10,))
        else:
            features = tf.parse_single_example(serialized_example,  # see parse_single_sequence_example for sequence example
                                               features={
                                                   'label': tf.FixedLenFeature([], tf.int64),
                                                   'img_raw' : tf.FixedLenFeature([], tf.string),
                                               })
            label = tf.cast(features['label'], tf.int32)
        # You can do more image distortion here for training data
        img = tf.decode_raw(features['img_raw'], tf.float32)
#        img = tf.reshape(img, [28 * 28 * 1])
        img.set_shape((784,))
        return img, label

    def __init__(self, params):
        '''
        Constructor
        '''
