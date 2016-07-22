'''
Created on Oct 28, 2015

@author: kashefy
'''
from nose.tools import assert_equals, assert_is_instance, \
    assert_list_equal, assert_in, assert_almost_equal, assert_true, \
    assert_raises
import os
import tempfile
import shutil
import numpy as np
import h5py
from balance import CLNAME_OTHER
import balance_hdf5 as bal

class TestBalanceClassCountHDF5:

    @classmethod
    def setup_class(self):

        self.dir_tmp = tempfile.mkdtemp()

        # generate fake data
        num_examples = 70
        num_classes = 2  # EXCLUDING other class
        cl = np.random.randint(0, num_classes, size=(num_examples))
        self.l = np.zeros((num_examples, num_classes))
        self.l[range(num_examples), cl] = 1
        self.l = np.vstack((self.l,
                            np.zeros((num_examples * 2, num_classes), dtype=self.l.dtype)))
        self.l = np.expand_dims(self.l, 1)
        self.l = np.expand_dims(self.l, -1)
        num_examples, _, num_classes, _ = self.l.shape
        self.x1 = np.random.randn(num_examples, 4, 3, 2)
        self.x2 = np.random.randn(num_examples, 1, 2, 3)

        self.fpath = os.path.join(self.dir_tmp, 'foo.h5')
        h = h5py.File(self.fpath, 'w')
        h['f1'] = self.x1
        h['f2'] = self.x2
        h['label'] = self.l
        h.close()

    @classmethod
    def teardown_class(self):

        shutil.rmtree(self.dir_tmp)

    def test_get_class_count_hdf5(self):

        counts = bal.get_class_count_hdf5(self.fpath,
                                          key_label='label',
                                          other_clname=CLNAME_OTHER)
        assert_is_instance(counts, dict, "Unexpected return instance type.")
        _, _, num_classes, _ = self.l.shape
        assert_equals(len(counts.keys()), num_classes + 1,
                      "Expecting a key for each class + 1 for 'other'.")

        assert_in(CLNAME_OTHER, counts.keys())

        for key in counts.keys():
            if key == CLNAME_OTHER:
                assert_equals(counts[key], 140,
                              "Unexpected count for '%s' class" % (CLNAME_OTHER,))
            else:
                assert_equals(counts[key], np.sum(self.l[:, :, int(key), :]),
                              "Unexpected count for class '%s'" % (key,))

    def test_balance_class_count_hdf5(self):

        dict_balanced, idxs = \
            bal.balance_class_count_hdf5(self.fpath,
                                         ['f1', 'f2'],
                                         key_label='label',
                                         other_clname=CLNAME_OTHER)
        assert_is_instance(dict_balanced, dict, "Unexpected return instance type.")
        keys_actual = list(dict_balanced.keys())
        keys_actual.sort()
        assert_list_equal(['f1', 'f2', 'label'],
                          keys_actual)

        for count, idx in enumerate(idxs):
            assert_true(np.all(self.l[idx] == dict_balanced['label'][count]))
            assert_true(np.all(self.x1[idx] == dict_balanced['f1'][count]))
            assert_true(np.all(self.x2[idx] == dict_balanced['f2'][count]))

        for cl in xrange(self.l.shape[2]):
            assert_almost_equal(np.count_nonzero(dict_balanced['label'][:, :, cl, :]),
                                140, 1)

    def test_save_balanced_class_count_hdf5(self):

        fpath_dst = os.path.join(self.dir_tmp, "save_balanced_class_count_dst.h5")

        idxs = \
            bal.save_balanced_class_count_hdf5(self.fpath,
                                               ['f1', 'f2'],
                                               fpath_dst)

        h = h5py.File(fpath_dst, 'r')
        keys_actual = list(h.keys())
        keys_actual.sort()
        assert_list_equal(['f1', 'f2', 'label'],
                          keys_actual)

        for count, idx in enumerate(idxs):
            assert_true(np.all(self.l[idx] == h['label'][count]))
            assert_true(np.all(self.x1[idx] == h['f1'][count]))
            assert_true(np.all(self.x2[idx] == h['f2'][count]))

        for cl in xrange(self.l.shape[2]):
            assert_almost_equal(np.count_nonzero(h['label'][:, :, cl, :]),
                                140, 1)

    def test_save_balanced_sampled_class_count_hdf5_target_count_max(self):

        fpath_dst = os.path.join(self.dir_tmp, "save_balanced_class_count_dst.h5")

        idxs = \
            bal.save_balanced_sampled_class_count_hdf5(self.fpath,
                                                       ['f1', 'f2'],
                                                       fpath_dst,
                                                       target_count=None)

        h = h5py.File(fpath_dst, 'r')
        keys_actual = list(h.keys())
        keys_actual.sort()
        assert_list_equal(['f1', 'f2', 'label'],
                          keys_actual)

        for count, idx in enumerate(idxs):
            assert_true(np.all(self.l[idx] == h['label'][count]))
            assert_true(np.all(self.x1[idx] == h['f1'][count]))
            assert_true(np.all(self.x2[idx] == h['f2'][count]))

        for cl in xrange(self.l.shape[2]):
            assert_almost_equal(np.count_nonzero(h['label'][:, :, cl, :]),
                                140, 1)

    def test_save_balanced_sampled_class_count_hdf5_target_count_lt_max(self):

        fpath_dst = os.path.join(self.dir_tmp, "save_balanced_class_count_dst.h5")

        idxs = \
            bal.save_balanced_sampled_class_count_hdf5(self.fpath,
                                                       ['f1', 'f2'],
                                                       fpath_dst,
                                                       target_count=99)

        h = h5py.File(fpath_dst, 'r')
        keys_actual = list(h.keys())
        keys_actual.sort()
        assert_list_equal(['f1', 'f2', 'label'],
                          keys_actual)

        for count, idx in enumerate(idxs):
            assert_true(np.all(self.l[idx] == h['label'][count]))
            assert_true(np.all(self.x1[idx] == h['f1'][count]))
            assert_true(np.all(self.x2[idx] == h['f2'][count]))

        for cl in xrange(self.l.shape[2]):
            assert_equals(np.count_nonzero(h['label'][:, :, cl, :]), 99)

    def test_save_balanced_sampled_class_count_hdf5_target_count_gt_max(self):

        fpath_dst = os.path.join(self.dir_tmp, "save_balanced_class_count_dst.h5")

        idxs = \
            bal.save_balanced_sampled_class_count_hdf5(self.fpath,
                                                       ['f1', 'f2'],
                                                       fpath_dst,
                                                       target_count=200)

        h = h5py.File(fpath_dst, 'r')
        keys_actual = list(h.keys())
        keys_actual.sort()
        assert_list_equal(['f1', 'f2', 'label'],
                          keys_actual)

        for count, idx in enumerate(idxs):
            assert_true(np.all(self.l[idx] == h['label'][count]))
            assert_true(np.all(self.x1[idx] == h['f1'][count]))
            assert_true(np.all(self.x2[idx] == h['f2'][count]))

        for cl in xrange(self.l.shape[2]):
            assert_equals(np.count_nonzero(h['label'][:, :, cl, :]), 200)

    def test_save_balanced_class_count_hdf5_same_file(self):

        assert_raises(IOError,
                      bal.save_balanced_class_count_hdf5,
                      self.fpath,
                      ['f1', 'f2'],
                      self.fpath,
                      key_label='label',
                      other_clname=CLNAME_OTHER)
