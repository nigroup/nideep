'''
Created on Apr 6, 2016

@author: marcenacp
'''
from nose.tools import assert_equal
import os
import tempfile
import shutil
import numpy as np
import h5py
import read_hdf5 as r

class TestNumEntriesHDF5:
    def setup(self):
        self.dir_tmp = tempfile.mkdtemp()

        # First hdf5 file with fake data
        self.path_hdf5_1 = os.path.join(self.dir_tmp, 'path_1.h5')
        f = h5py.File(self.path_hdf5_1, 'w')
        # First dataset 'label'
        dset1 = f.create_dataset('label', (1000, 10), dtype='i')
        dset1[...] = np.arange(10000).reshape((1000,10))
        # Second dataset 'data'
        dset2 = f.create_dataset('data', (200,), dtype='i')
        dset2[...] = np.arange(200)

        f.close()

        # Second hdf5 file with fake data
        self.path_hdf5_2 = os.path.join(self.dir_tmp, 'path_2.h5')
        f = h5py.File(self.path_hdf5_2, 'w')
        # First dataset 'label'
        dset1 = f.create_dataset('label', (2000, 10), dtype='i')
        dset1[...] = np.arange(20000).reshape((2000,10))
        # Second dataset 'data'
        dset2 = f.create_dataset('data', (100,), dtype='i')
        dset2[...] = np.arange(100)

        f.close()

        # A *.txt file points to the two hdf5 files
        self.path_two_hdf5 = os.path.join(self.dir_tmp, 'path_two.txt')
        f = open(self.path_two_hdf5, 'w')
        f.write(self.path_hdf5_1+'\n')
        f.write(self.path_hdf5_2+'\n')

        f.close()

        # Another *.txt file points to one hdf5 file only
        self.path_one_hdf5 = os.path.join(self.dir_tmp, 'path_one.txt')
        f = open(self.path_one_hdf5, 'w')
        f.write(self.path_hdf5_1+'\n')

        f.close()

    def teardown(self):
        shutil.rmtree(self.dir_tmp)

    def test_num_entries(self):
        assert_equal(3000, r.num_entries(self.path_two_hdf5, 'label'))
        assert_equal(1000, r.num_entries(self.path_one_hdf5, 'label'))
        assert_equal(300, r.num_entries(self.path_two_hdf5, 'data'))
        assert_equal(200, r.num_entries(self.path_one_hdf5, 'data'))
