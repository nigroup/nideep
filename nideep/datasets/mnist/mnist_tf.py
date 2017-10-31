'''
Created on Jul 31, 2017

@author: kashefy
'''
import os
from collections import namedtuple
import numpy as np
import skimage.transform as transform
from tensorflow.python.framework import dtypes, random_seed
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet, read_data_sets
import tensorflow as tf
# imported for mocking
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images
from _ast import Num

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
    def to_tf_record(cls,
                     fpath,
                     data_dir,
                     fake_data=False,
                     one_hot=False,
                     dtype=dtypes.float32,
                     reshape=True,
                     validation_size=5000,
                     seed=None,
                     orientations=np.linspace(-90, 90, 180/(12+1), endpoint=True).tolist(),
                    ):
        """
        Get dataset with orientations stored in tf.records
        """
        mnist = MNIST.read_data_sets(
                                     data_dir,
                                     fake_data=fake_data,
                                     one_hot=one_hot,
                                     dtype=dtype,
                                     reshape=False,
                                     validation_size=validation_size,
                                     seed=seed,
                                     )
        fname0, ext = os.path.splitext(fpath)
        DatasetTFRecords = namedtuple('DatasetTFRecords',
                                      ['num_examples',
                                       'path',
                                       'one_hot',
                                       'orientations',
                                       'phase',
                                       'images',
                                       'labels'])
        dsets = {}
        for phase in Datasets._fields:#['train', 'validation', 'test']:
            split = getattr(mnist, phase)
            num_examples = 0
            if fname0.endswith(phase):
                fpath_phase = fpath
            else:
                fpath_phase = ''.join([fname0, '_', phase, ext])
            if os.path.isfile(fpath_phase):
                # Skip if it already exists
                print(fpath_phase)
                num_examples = sum(1 for _ in tf.python_io.tf_record_iterator(fpath_phase))
            else:
                with tf.python_io.TFRecordWriter(fpath_phase) as writer:
                    for img, label in zip(split.images, split.labels):
                        if img.ndim < 2:
                            raise AttributeError("Rotation needs both height and width images resolved, found shape %s" % img.shape) 
                        img = np.expand_dims(img, axis=0)
                        imgs_orient_all, labels_orient_all = \
                            MNIST.rotate(img, orientations, one_hot)
                        for img_orient, label_orient in zip(imgs_orient_all, labels_orient_all):
                            img_raw = img_orient.tobytes()
                            ## Convert the bytes back to image as follow:
                            # image = Image.frombytes('RGB', (224,224), img_raw)
                            # tl.visualize.frame(I=image, second=1, saveable=False, name='frame', fig_idx=1236)
                            ## Write the data into TF format
                            # image     : Feature + BytesList
                            # label     : Feature + Int64List or FloatList
                            # sentence  : FeatureList + Int64List , see Google's im2txt example
                            if one_hot:
                                label_raw = label.astype(np.float32).tobytes()
                                label_orient_raw = label_orient.astype(np.float32).tobytes()
                                example = tf.train.Example(features=tf.train.Features(feature={
                                    "label"         : _bytes_feature(label_raw),
                                    "label_orient"  : _bytes_feature(label_orient_raw),
                                    'img_raw'       : _bytes_feature(img_raw),
                                    }))
                            else:
                                example = tf.train.Example(features=tf.train.Features(feature={
                                    "label"         : _int64_feature(int(label)),
                                    "label_orient"  : _int64_feature(int(label_orient)),
                                    'img_raw'       : _bytes_feature(img_raw),
                                    }))
                            writer.write(example.SerializeToString())  # Serialize To String
                            num_examples += 1
            dsets[phase] = DatasetTFRecords(
                                num_examples=num_examples,
                                path=fpath_phase,
                                one_hot=one_hot,
                                orientations=orientations,
                                phase=phase,
                                images=np.empty((num_examples,) + split.images.shape[1:],
                                                dtype=split.images.dtype),
                                labels=np.empty((num_examples,) + split.labels.shape[1:],
                                                dtype=split.labels.dtype),
                                )
        return Datasets(train=dsets['train'],
                        validation=dsets['validation'],
                        test=dsets['test'])
                    
    @classmethod        
    def read_and_decode_ops(cls, fpath, one_hot=False,
                            img_shape=(784,), # or (28, 28, 1)
                            num_classes=10,
                            num_orientations=0,
                            ):
        # generate a queue with a given file name
        filename_queue = tf.train.string_input_producer([fpath])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)     # return the file and the name of file
        feat_dict = {
            'label'     : tf.FixedLenFeature([], [tf.int64, tf.string][one_hot]),
            'img_raw'   : tf.FixedLenFeature([], tf.string),
            }
        if num_orientations > 0:
            feat_dict['label_orient'] = tf.FixedLenFeature([], [tf.int64, tf.string][one_hot])
        features = tf.parse_single_example(serialized_example,  # see parse_single_sequence_example for sequence example
                                           features=feat_dict)
        if one_hot:
            label = tf.decode_raw(features['label'], tf.float32)            
#            label = tf.reshape(label, [10])
            label.set_shape((num_classes,))
            if num_orientations > 0:
                label_orient = tf.decode_raw(features['label_orient'], tf.float32)            
    #            label = tf.reshape(label, [10])           
                label_orient.set_shape((num_orientations,))
        else:
            label = tf.cast(features['label'], tf.int32)
            if num_orientations > 0:
                label_orient = tf.cast(features['label_orient'], tf.int32)
        # You can do more image distortion here for training data
        img = tf.decode_raw(features['img_raw'], tf.float32)
#        img = tf.reshape(img, [28 * 28 * 1])
        img.set_shape((784,))
        if num_orientations > 0:
            return img, label, label_orient
        else:
            return img, label
    
    @staticmethod
    def rotate(imgs,
               orientations,
               one_hot,
               mode='symmetric',
               cval=0):
        num_examples_in = len(imgs)
        labels_orient_all = None
        imgs_orient_all = None
        for angle_idx, angle in enumerate(orientations):
            imgs_orient = np.zeros_like(imgs)
            for img_idx, img in enumerate(imgs):
                img_rot = transform.rotate(img, angle,
                                           resize=False, center=None,
                                           order=1,
                                           mode=mode, cval=cval,
                                           clip=True, preserve_range=False)
                imgs_orient[img_idx] = img_rot
            if one_hot:
                labels_orient = np.zeros((num_examples_in, len(orientations)))
                labels_orient[:, angle_idx] = 1
            else:
                labels_orient = np.zeros((num_examples_in,)) + angle_idx
            if labels_orient_all is None:
                labels_orient_all = labels_orient
                imgs_orient_all = imgs_orient
            else:
                labels_orient_all = np.vstack((labels_orient_all, labels_orient))
                imgs_orient_all = np.vstack((imgs_orient_all, imgs_orient))
        return imgs_orient_all, labels_orient_all
            
    @classmethod
    def read_data_sets_orient(cls,
                              train_dir,
                              fake_data=False,
                              one_hot=False,
                              dtype=dtypes.float32,
                              reshape=True,
                              validation_size=5000,
                              seed=None,
                              orientations=np.linspace(-90, 90, 180/(12+1), endpoint=True).tolist()):
        
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        np.random.seed(seed1 if seed is None else seed2)
        ds = read_data_sets(\
                                  train_dir,
                                  fake_data=fake_data,
                                  one_hot=one_hot,
                                  dtype=dtype,
                                  reshape=False, # preseve image dimensions for rotation
                                  validation_size=0,
                                  seed=seed)
        imgs_orient_all, labels_orient_all = MNIST.rotate(ds.train.images, orientations, one_hot)
        print(labels_orient_all.shape, imgs_orient_all.shape)
        # An image and all its orientation either fall into train or validation
        perm0 = np.arange(ds.train.num_examples)
        np.random.shuffle(perm0)
        train_idxs0 = perm0[validation_size:]
        val_idxs0 = perm0[:validation_size]
        train_idxs = train_idxs0
        val_idxs = val_idxs0
        for orient_idx in range(len(orientations)):
            train_idxs_orient = train_idxs0 + (orient_idx+1)*ds.train.num_examples
            val_idxs_orient = val_idxs0 + (orient_idx+1)*ds.train.num_examples
            train_idxs = np.vstack((train_idxs, train_idxs_orient))
            val_idxs = np.vstack((val_idxs, val_idxs_orient))
        np.random.shuffle(train_idxs)
        np.random.shuffle(val_idxs)
        train = DataSet(\
                    np.multiply(imgs_orient_all[train_idxs], [1., 255.][dtype == dtypes.float32]), # will rescale to [0,1] inside
                    labels_orient_all[train_idxs],
                    fake_data=fake_data,
                    one_hot=one_hot,
                    dtype=dtype,
                    reshape=reshape, # reshape here for the first time 
                    seed=seed)
        validation = DataSet(\
                        np.multiply(imgs_orient_all[val_idxs], [1., 255.][dtype == dtypes.float32]), # will rescale to [0,1] inside
                        labels_orient_all[val_idxs],
                        fake_data=fake_data,
                        one_hot=one_hot,
                        dtype=dtype,
                        reshape=reshape, # reshape here for the first time 
                        seed=seed)
        validation = DataSet(\
                        np.multiply(imgs_orient_all[val_idxs], [1., 255.][dtype == dtypes.float32]), # will rescale to [0,1] inside
                        labels_orient_all[val_idxs],
                        fake_data=fake_data,
                        one_hot=one_hot,
                        dtype=dtype,
                        reshape=reshape, # reshape here for the first time 
                        seed=seed)
        labels_orient_all, imgs_orient_all = MNIST.rotate(ds.test.images, orientations, one_hot)
        test = DataSet(\
                        np.multiply(imgs_orient_all, [1., 255.][dtype == dtypes.float32]), # will rescale to [0,1] inside
                        labels_orient_all,
                        fake_data=fake_data,
                        one_hot=one_hot,
                        dtype=dtype,
                        reshape=reshape, # reshape here for the first time 
                        seed=seed)
        return Datasets(train=train, validation=validation, test=ds.test)

    def __init__(self, params):
        '''
        Constructor
        '''
