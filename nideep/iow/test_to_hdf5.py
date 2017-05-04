'''
Created on Jan 20, 2016

@author: kashefy
'''
from nose.tools import assert_equal, assert_true, assert_list_equal, \
    assert_is_instance, assert_equals, assert_in, assert_raises
import os
import tempfile
import shutil
import numpy as np
import h5py
import to_hdf5 as to

class TestArraysToHDF5:

    @classmethod
    def setup_class(self):

        self.dir_tmp = tempfile.mkdtemp()

        x = np.array([[[ 1, 2, 3],
                       [ 4, 5, 6]
                       ],
                      [[ 7, 8, 9],
                       [10, 11, 12]
                       ],
                      [[13, 14, 15],
                       [16, 17, 18],
                       ],
                      [[19, 20, 21],
                       [22, 23, 24]
                       ]
                      ])

        self.arr = [x, x + 1]

    @classmethod
    def teardown_class(self):

        shutil.rmtree(self.dir_tmp)

    def test_arr_single(self):

        # use the module and test it
        fpath = os.path.join(self.dir_tmp, 'xarr1.h5')
        to.arrays_to_h5_fixed([self.arr[0]], 'x', fpath)

        with h5py.File(fpath, 'r') as h:
            assert_list_equal(h.keys(), ['x'])
            assert_equal(1, len(h['x']))
            assert_true(np.all(self.arr[0] == h['x'][:]))

    def test_arr(self):

        fpath = os.path.join(self.dir_tmp, 'xarr.h5')
        to.arrays_to_h5_fixed(self.arr, 'x', fpath)

        with h5py.File(fpath, 'r') as h:
            assert_list_equal(h.keys(), ['x'])
            assert_equal(2, len(h['x']))
            for x, y in zip(self.arr, h['x'][:]):
                assert_true(np.all(x == y))

    def test_arr_shape(self):

        fpath = os.path.join(self.dir_tmp, 'xarr_sh.h5')
        to.arrays_to_h5_fixed(self.arr, 'x', fpath)

        with h5py.File(fpath, 'r') as h:
            assert_list_equal(h.keys(), ['x'])
            assert_equal(2, len(h['x']))
            for x, y in zip(self.arr, h['x'][:]):
                assert_equal(x.shape, y.shape)

class TestSplitHDF5:

    def setup(self):

        self.dir_tmp = tempfile.mkdtemp()

        x = np.array([[[ 1, 2, 3],
                       [ 4, 5, 6]
                       ],
                      [[ 7, 8, 9],
                       [10, 11, 12]
                       ],
                      [[13, 14, 15],
                       [16, 17, 18],
                       ],
                      [[19, 20, 21],
                       [22, 23, 24]
                       ]
                      ])
        self.arr = [x, x + 100, x + 1000, x + 10000, x + 100000]
        self.fpath = os.path.join(self.dir_tmp, 'foo.h5')

        with h5py.File(self.fpath, 'w') as f:
            f['x1'] = [x for x in self.arr]
            f['x2'] = [x + 0.1 for x in self.arr]

    def teardown(self):

        shutil.rmtree(self.dir_tmp)

    def test_split(self):

        h5_list = to.split_hdf5(self.fpath, self.dir_tmp, tot_floats=((3 * 4 * 2 * 3)))
        # hdf5_list = to.split_hdf5(self.fpath, self.dir_tmp, tot_floats=((10*4*2*3)))
        # hdf5_list = to.split_hdf5(self.fpath, self.dir_tmp, tot_floats=((1*4*2*3)))

        assert_is_instance(h5_list, list)
        assert_equals(len(h5_list), 2)

        name_, ext = os.path.splitext(os.path.basename(self.fpath))

        for p in h5_list:
            assert_in(name_, p)
            assert_true(p.endswith(ext), "Unexpected extension")

        offset = 0
        with h5py.File(self.fpath, 'r') as h_src:
            for p in h5_list:
                with h5py.File(p, 'r') as h:
                    assert_list_equal(['x1', 'x2'], h.keys())

                    for k in h.keys():
                        min_len = min(len(h[k]), len(h_src[k]))
                        sub_actual = h[k][0:min_len]
                        sub_expected = h_src[k][offset:offset + min_len]
                        assert_true(np.all(sub_actual == sub_expected))

                    offset += min_len

    def test_split_valid_arg(self):

        assert_raises(IOError, to.split_hdf5, self.fpath, 'foo/')
