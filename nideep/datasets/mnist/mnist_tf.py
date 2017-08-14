'''
Created on Jul 31, 2017

@author: kashefy
'''
import numpy as np
from tensorflow.python.framework import dtypes, random_seed
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet, read_data_sets
# imported for mocking
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images

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
                    ds.train.images[train_idxs],
                    ds.train.labels[train_idxs],
                    fake_data=fake_data,
                    one_hot=one_hot,
                    dtype=dtype,
                    reshape=False, # already reshaped
                    seed=seed)
        validation = DataSet(\
                        ds.train.images[val_idxs],
                        ds.train.labels[val_idxs],
                        fake_data=fake_data,
                        one_hot=one_hot,
                        dtype=dtype,
                        reshape=False, # already reshaped
                        seed=seed)
        return Datasets(train=train, validation=validation, test=ds.test)

    def __init__(self, params):
        '''
        Constructor
        '''
