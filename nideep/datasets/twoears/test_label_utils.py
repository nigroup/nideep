'''
Created on Apr 25, 2017

@author: kashefy
'''
import os
import tempfile
import shutil
import numpy as np
from nose.tools import assert_true, \
    assert_list_equal, assert_not_in, assert_in, assert_equal
from h5py import File
import label_utils as lu

class TestLabelUtils:

    @classmethod
    def setup_class(self):

        self.dir_tmp = tempfile.mkdtemp()

    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.dir_tmp)

    def test_id_loc_to_loc(self):

        p = os.path.join(self.dir_tmp, 'x.h5')
        new_key = 'new_key'
        fake_data = np.array([[[[  1, 2, 3]],
                               [[ 10, 20, 30]],
                               [[100, 200, 300]]],
                              [[[  4, 5, 6]],
                               [[ 40, 50, 60]],
                               [[400, 500, 600]]]])
        with File(p, 'w') as h:
            h['label_id_loc'] = fake_data
            assert_not_in(new_key, h)
            lu.id_loc_to_loc(p, new_key)
            assert_in(new_key, h, "new key missing")

        with File(p, 'r') as h:
            x = h[new_key][:]
            assert_equal(x.ndim, fake_data.ndim)
            assert_equal(x.shape[-1], fake_data.shape[-1] - 1, "void bin expected to be eliminated")
            assert_equal(x.shape[0], fake_data.shape[0], "batch bin has changed")
            expected = \
                np.array([[[[111, 222, 333]]],
                          [[[444, 555, 666]]]])
            assert_true(np.all(x == expected[..., :-1]))
        pass

    def test_id_loc_to_loc_incl_void(self):

        p = os.path.join(self.dir_tmp, 'x.h5')
        new_key = 'new_key'
        fake_data = np.array([[[[  1, 2, 3]],
                               [[ 10, 20, 30]],
                               [[100, 200, 300]]],
                              [[[  4, 5, 6]],
                               [[ 40, 50, 60]],
                               [[400, 500, 600]]]])
        with File(p, 'w') as h:
            h['label_id_loc'] = fake_data
            assert_not_in(new_key, h)
            lu.id_loc_to_loc(p, new_key, has_void_bin=False)
            assert_in(new_key, h, "new key missing")

        with File(p, 'r') as h:
            x = h[new_key][:]
            assert_equal(x.ndim, fake_data.ndim)
            assert_equal(x.shape[-1], fake_data.shape[-1], "void bin expected to be preserved")
            assert_equal(x.shape[0], fake_data.shape[0], "batch bin has changed")
            expected = \
                np.array([[[[111, 222, 333]]],
                          [[[444, 555, 666]]]])
            assert_true(np.all(x == expected))
        pass

    def test_id_loc_to_loc_keys(self):

        p = os.path.join(self.dir_tmp, 'x.h5')
        fake_data = np.array([[[[  1, 2, 3]],
                       [[ 10, 20, 30]],
                       [[100, 200, 300]]],
                      [[[  4, 5, 6]],
                       [[ 40, 50, 60]],
                       [[400, 500, 600]]]])
        with File(p, 'w') as h:
            h['label_id_loc'] = fake_data

            keys = [k.encode('ascii', 'ignore') for k in h.keys()]
            for new_key in ['a', 'b', 'new_key']:
                assert_not_in(new_key, h)
                keys.append(new_key)
                keys.sort()
                sorted(keys)
                lu.id_loc_to_loc(p, new_key)
                assert_in(new_key, h, "new key missing")
                k2 = [k.encode('ascii', 'ignore') for k in h.keys()]
                sorted(k2)
                k2.sort()
                assert_list_equal(keys, k2)
        pass

