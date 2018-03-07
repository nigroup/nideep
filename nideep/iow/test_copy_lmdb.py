'''
Created on Oct 30, 2015

@author: kashefy
'''
from nose.tools import assert_equal, assert_false, \
    assert_true, assert_greater
import os
import tempfile
import shutil
import numpy as np
import lmdb
import caffe
from nideep.iow.lmdb_utils import IDX_FMT, MAP_SZ
import nideep.iow.to_lmdb as tol
import nideep.iow.read_lmdb as r
import nideep.iow.copy_lmdb as c

class TestCopySamplesLMDB:

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
        tol.arrays_to_lmdb([y for y in x], os.path.join(self.dir_tmp, 'x_lmdb'))

    @classmethod
    def teardown_class(self):

        shutil.rmtree(self.dir_tmp)

    def test_copy_samples_single(self):

        path_src = os.path.join(self.dir_tmp, 'x_lmdb')
        x = r.read_values(path_src)
        path_dst = os.path.join(self.dir_tmp, 'test_copy_samples_single_lmdb')
        assert_greater(len(x), 0, "This test needs non empty data.")
        for i in xrange(len(x)):
            if os.path.isdir(path_dst):
                shutil.rmtree(path_dst)
            c.copy_samples_lmdb(path_src, path_dst, [i])
            assert_true(os.path.isdir(path_dst), "failed to save LMDB for i=%d" % (i,))

            y = r.read_values(path_dst)
            assert_equal(len(y), 1, "Single element expected.")
            assert_true(np.all(x[i][0] == y[0][0]), "Wrong content copied.")
            assert_true(np.all(x[i][1] == y[0][1]), "Wrong content copied.")

    def test_copy_samples_single_with_data_func(self):

        path_src = os.path.join(self.dir_tmp, 'x_lmdb')
        x = r.read_values(path_src)
        path_dst = os.path.join(self.dir_tmp, 'test_copy_samples_single_with_data_func_lmdb')
        assert_greater(len(x), 0, "This test needs non empty data.")
        ALPHA = -10  # non-zero
        def func_data_mul(value):
            _, v = r.unpack_raw_datum(value)
            dat = caffe.io.array_to_datum(ALPHA * v, int(v.flatten()[0]))
            return dat.SerializeToString()

        for i in xrange(len(x)):
            if os.path.isdir(path_dst):
                shutil.rmtree(path_dst)
            c.copy_samples_lmdb(path_src, path_dst, [i], func_data=func_data_mul)
            assert_true(os.path.isdir(path_dst), "failed to save LMDB for i=%d" % (i,))

            y = r.read_values(path_dst)
            assert_equal(len(y), 1, "Single element expected.")
            assert_true(np.all(x[i][0] * ALPHA == y[0][0]), "Wrong content copied.")
            assert_true(np.all(int(x[i][0].flatten()[0]) == y[0][1]), "Wrong content copied for label.")

    def test_copy_samples_single_reverse(self):

        path_src = os.path.join(self.dir_tmp, 'x_lmdb')
        x = r.read_values(path_src)
        path_dst = os.path.join(self.dir_tmp, 'test_copy_samples_single_lmdb')
        assert_greater(len(x), 0, "This test needs non empty data.")
        for i in range(len(x))[::-1]:
            if os.path.isdir(path_dst):
                shutil.rmtree(path_dst)
            c.copy_samples_lmdb(path_src, path_dst, [i])
            assert_true(os.path.isdir(path_dst), "failed to save LMDB for i=%d" % (i,))

            y = r.read_values(path_dst)
            assert_equal(len(y), 1, "Single element expected.")
            assert_true(np.all(x[i][0] == y[0][0]), "Wrong content copied.")
            assert_true(np.all(x[i][1] == y[0][1]), "Wrong content copied.")

    def test_copy_samples_subset(self):

        path_src = os.path.join(self.dir_tmp, 'x_lmdb')
        x = r.read_values(path_src)
        assert_greater(len(x), 0, "This test needs non empty data.")
        path_dst = os.path.join(self.dir_tmp, 'test_copy_samples_subset')
        keys = range(0, len(x), 2)
        assert_greater(len(keys), 0, "This test needs a non-empty subset.")
        assert_greater(len(x), len(keys), "Need subset, not all elements.")
        c.copy_samples_lmdb(path_src, path_dst, keys)
        assert_true(os.path.isdir(path_dst), "failed to save LMDB")

        y = r.read_values(path_dst)
        assert_equal(int(len(x) / 2), len(y), "Wrong number of elements copied.")
        for (x_val, x_label), (y_val, y_label) in zip(x[0::2], y):  # skip element in x
            assert_true(np.all(x_val == y_val), "Wrong content copied.")
            assert_true(np.all(x_label == y_label), "Wrong content copied.")

    def test_copy_samples_subset_with_data_func(self):

        path_src = os.path.join(self.dir_tmp, 'x_lmdb')
        x = r.read_values(path_src)
        assert_greater(len(x), 0, "This test needs non empty data.")
        path_dst = os.path.join(self.dir_tmp, 'test_copy_samples_subset_with_data_func_lmdb')
        keys = range(0, len(x), 2)
        assert_greater(len(keys), 0, "This test needs a non-empty subset.")
        assert_greater(len(x), len(keys), "Need subset, not all elements.")

        ALPHA = -10  # non-zero
        def func_data_mul(value):
            _, x = r.unpack_raw_datum(value)
            dat = caffe.io.array_to_datum(ALPHA * x, int(x.flatten()[0]))
            return dat.SerializeToString()

        c.copy_samples_lmdb(path_src, path_dst, keys, func_data=func_data_mul)
        assert_true(os.path.isdir(path_dst), "failed to save LMDB")

        y = r.read_values(path_dst)
        assert_equal(int(len(x) / 2), len(y), "Wrong number of elements copied.")
        for (x_val, x_label), (y_val, y_label) in zip(x[0::2], y):  # skip element in x
            assert_true(np.all(x_val * ALPHA == y_val), "Wrong content copied.")
            assert_true(np.all(int(x_val.flatten()[0]) == y_label), "Wrong content copied for label.")
            assert_false(np.all(x_label == y_label), "Wrong content copied for label.")

    def test_copy_samples_no_append(self):

        path_src = os.path.join(self.dir_tmp, 'x_lmdb')
        x = r.read_values(path_src)
        path_dst = os.path.join(self.dir_tmp, 'test_copy_samples_subset')
        c.copy_samples_lmdb(path_src, path_dst, range(0, len(x), 2))
        c.copy_samples_lmdb(path_src, path_dst, range(0, len(x), 2))
        c.copy_samples_lmdb(path_src, path_dst, range(0, len(x), 2))
        c.copy_samples_lmdb(path_src, path_dst, range(0, len(x), 2))
        assert_true(os.path.isdir(path_dst), "failed to save LMDB")

        y = r.read_values(path_dst)
        assert_equal(int(len(x) / 2), len(y), "Wrong number of elements copied.")
        for a, b in zip(x[0::2], y):  # skip element in x
            assert_true(np.all(a[0] == b[0]), "Wrong content copied.")
            assert_true(np.all(a[1] == b[1]), "Wrong content copied.")

    def test_copy_samples_all(self):

        path_src = os.path.join(self.dir_tmp, 'x_lmdb')
        x = r.read_values(path_src)
        path_dst = os.path.join(self.dir_tmp, 'test_copy_samples_all_lmdb')
        c.copy_samples_lmdb(path_src, path_dst, range(len(x)))
        assert_true(os.path.isdir(path_dst), "failed to save LMDB")

        y = r.read_values(path_dst)
        assert_equal(len(x), len(y), "Wrong number of elements copied.")
        for a, b in zip(x, y):
            assert_true(np.all(a[0] == b[0]), "Wrong content copied.")
            assert_true(np.all(a[1] == b[1]), "Wrong content copied.")

class TestConcatentateLMDB:

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

        tol.arrays_to_lmdb([y for y in x], os.path.join(self.dir_tmp, 'x0_lmdb'))
        tol.arrays_to_lmdb([-y for y in x], os.path.join(self.dir_tmp, 'x1_lmdb'))
        tol.arrays_to_lmdb([y + 1000 for y in x], os.path.join(self.dir_tmp, 'x2_lmdb'))

    @classmethod
    def teardown_class(self):

        shutil.rmtree(self.dir_tmp)

    def test_concatenate(self):

        path_src0 = os.path.join(self.dir_tmp, 'x0_lmdb')
        x0 = r.read_values(path_src0)
        path_src1 = os.path.join(self.dir_tmp, 'x1_lmdb')
        x1 = r.read_values(path_src1)
        path_src2 = os.path.join(self.dir_tmp, 'x2_lmdb')
        x2 = r.read_values(path_src2)
        path_dst = os.path.join(self.dir_tmp, 'test_concatenate_lmdb')
        c.concatenate_lmdb([path_src0, path_src1, path_src2], path_dst)
        assert_true(os.path.isdir(path_dst), "failed to save LMDB")

        y = r.read_values(path_dst)
        assert_equal(len(x0) + len(x1) + len(x2), len(y), "Wrong number of elements copied.")
        x = x0
        x.extend(x1)
        x.extend(x2)
        assert_equal(len(x), len(y), "Wrong number of elements copied.")
        for a, b in zip(x, y):
            assert_true(np.all(a[0] == b[0]), "Wrong content copied.")
            assert_true(np.all(a[1] == b[1]), "Wrong content copied.")
