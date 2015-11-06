'''
Created on Oct 30, 2015

@author: kashefy
'''
from nose.tools import assert_equal, assert_true, assert_raises
from mock import patch, MagicMock
import os
import tempfile
import shutil
import numpy as np
from scipy import io
import lmdb
import cv2 as cv2
import to_lmdb as tol
import caffe

class DatumMock:
    
    def SerializeToString(self):
        return
        
class TestImagesToLMDB:

    @classmethod
    def setup_class(self):
        
        self.dir_tmp = tempfile.mkdtemp()
        
        x = np.array([[[ 1,  2,  3],
                       [ 4,  5,  6]
                       ],
                      [[ 7,  8,  9],
                       [10, 11, 12]
                       ],
                      [[13, 14, 15],
                       [16, 17, 18],
                       ],
                      [[19, 20, 21],
                       [22, 23, 24]
                       ]
                      ])
        
        self.path_img1 = os.path.join(self.dir_tmp, "a.png")
        cv2.imwrite(self.path_img1, x)
        
        self.path_img2 = os.path.join(self.dir_tmp, "b.png")
        cv2.imwrite(self.path_img2, x+1)
        
    @classmethod
    def teardown_class(self):
        
        shutil.rmtree(self.dir_tmp)
            
    @patch('to_lmdb.caffe')
    @patch('to_lmdb.caffe.proto.caffe_pb2.Datum')
    def test_img_str(self, mock_dat, mock_caffe):
        
        # expected serialization of the test image
        s = '\x08\x03\x10\x04\x18\x02"\x18\x01\x04\x07\n\r\x10\x13\x16\x02\x05\x08\x0b\x0e\x11\x14\x17\x03\x06\t\x0c\x0f\x12\x15\x18(\x00'
        
        # mock caffe calls made by our module
        mock_dat.return_value.SerializeToString.return_value = s
        mock_caffe.io.array_to_datum.return_value = caffe.proto.caffe_pb2.Datum()
        
        # use the module and test it
        path_lmdb = os.path.join(self.dir_tmp, 'x1_lmdb')
        tol.imgs_to_lmdb([self.path_img1], path_lmdb)
        assert_true(os.path.isdir(path_lmdb), "failed to save LMDB")
        
        env_src = lmdb.open(path_lmdb, readonly=True)
        
        count = 0
        
        with env_src.begin() as txn:
            
            cursor = txn.cursor()
            for key, value in cursor:
                
                assert_equal(key, '0000000000', "Unexpected key.")
                assert_equal(value, s, "Unexpected content.")
                
                count += 1
        
        assert_equal(count, 1, "Unexpected number of samples.")   
            
    @patch('to_lmdb.caffe')
    @patch('to_lmdb.caffe.proto.caffe_pb2.Datum')
    def test_imgs_str(self, mock_dat, mock_caffe):
        
        # expected serialization of the test image
        s = ['\x08\x03\x10\x04\x18\x02"\x18\x01\x04\x07\n\r\x10\x13\x16\x02\x05\x08\x0b\x0e\x11\x14\x17\x03\x06\t\x0c\x0f\x12\x15\x18(\x00',
             '\x08\x03\x10\x04\x18\x02"\x18\x02\x05\x08\x0b\x0e\x11\x14\x17\x03\x06\t\x0c\x0f\x12\x15\x18\x04\x07\n\r\x10\x13\x16\x19(\x00',
             ]
        
        # mock caffe calls made by our module
        mock_dat.return_value.SerializeToString = MagicMock(side_effect=s)
        mock_caffe.io.array_to_datum.return_value = caffe.proto.caffe_pb2.Datum()
        
        # use the module and test it
        path_lmdb = os.path.join(self.dir_tmp, 'x2_lmdb')
        tol.imgs_to_lmdb([self.path_img1, self.path_img2], path_lmdb)
        assert_true(os.path.isdir(path_lmdb), "failed to save LMDB")
        
        env_src = lmdb.open(path_lmdb, readonly=True)
        
        count = 0
        
        with env_src.begin() as txn:
            
            cursor = txn.cursor()
            for key, value in cursor:
                
                k = tol.IDX_FMT.format(count)
                #print(k, value)
                #print(k, s[count])
                assert_equal(key, k, "Unexpected key.")
                assert_equal(value, s[count], "Unexpected content.")
                
                count += 1
        
        assert_equal(count, 2, "Unexpected number of samples.")
        
class TestMatFilesToLMDB:

    @classmethod
    def setup_class(self):
        
        self.dir_tmp = tempfile.mkdtemp()
        
        x = np.array([[[ 1,  2,  3],
                       [ 4,  5,  6]
                       ],
                      [[ 7,  8,  9],
                       [10, 11, 12]
                       ],
                      [[13, 14, 15],
                       [16, 17, 18],
                       ],
                      [[19, 20, 21],
                       [22, 23, 24]
                       ]
                      ])
        
        self.path_img1 = os.path.join(self.dir_tmp, "a.mat")
        io.savemat(self.path_img1, {'key': x})
        
        self.path_img2 = os.path.join(self.dir_tmp, "b.mat")
        io.savemat(self.path_img2, {'key': x+1})
        
    @classmethod
    def teardown_class(self):
        
        shutil.rmtree(self.dir_tmp)
            
    @patch('to_lmdb.caffe')
    @patch('to_lmdb.caffe.proto.caffe_pb2.Datum')
    def test_matfile_str(self, mock_dat, mock_caffe):
        
        # expected serialization of the test image
        s = '\x08\x03\x10\x04\x18\x02"\x18\x01\x04\x07\n\r\x10\x13\x16\x02\x05\x08\x0b\x0e\x11\x14\x17\x03\x06\t\x0c\x0f\x12\x15\x18(\x00'
        
        # mock caffe calls made by our module
        mock_dat.return_value.SerializeToString.return_value = s
        mock_caffe.io.array_to_datum.return_value = caffe.proto.caffe_pb2.Datum()
        
        # use the module and test it
        path_lmdb = os.path.join(self.dir_tmp, 'x1_lmdb')
        tol.matfiles_to_lmdb([self.path_img1], path_lmdb, 'key')
        assert_true(os.path.isdir(path_lmdb), "failed to save LMDB")
        
        env_src = lmdb.open(path_lmdb, readonly=True)
        
        count = 0
        
        with env_src.begin() as txn:
            
            cursor = txn.cursor()
            for key, value in cursor:
                
                assert_equal(key, '0000000000', "Unexpected key.")
                assert_equal(value, s, "Unexpected content.")
                
                count += 1
        
        assert_equal(count, 1, "Unexpected number of samples.")   
            
    @patch('to_lmdb.caffe')
    @patch('to_lmdb.caffe.proto.caffe_pb2.Datum')
    def test_matfiles_str(self, mock_dat, mock_caffe):
        
        # expected serialization of the test image
        s = ['\x08\x03\x10\x04\x18\x02"\x18\x01\x04\x07\n\r\x10\x13\x16\x02\x05\x08\x0b\x0e\x11\x14\x17\x03\x06\t\x0c\x0f\x12\x15\x18(\x00',
             '\x08\x03\x10\x04\x18\x02"\x18\x02\x05\x08\x0b\x0e\x11\x14\x17\x03\x06\t\x0c\x0f\x12\x15\x18\x04\x07\n\r\x10\x13\x16\x19(\x00',
             ]
        
        # mock caffe calls made by our module
        mock_dat.return_value.SerializeToString = MagicMock(side_effect=s)
        mock_caffe.io.array_to_datum.return_value = caffe.proto.caffe_pb2.Datum()
        
        # use the module and test it
        path_lmdb = os.path.join(self.dir_tmp, 'x2_lmdb')
        tol.matfiles_to_lmdb([self.path_img1, self.path_img2], path_lmdb, 'key')
        assert_true(os.path.isdir(path_lmdb), "failed to save LMDB")
        
        env_src = lmdb.open(path_lmdb, readonly=True)
        
        count = 0
        
        with env_src.begin() as txn:
            
            cursor = txn.cursor()
            for key, value in cursor:
                
                k = tol.IDX_FMT.format(count)
                #print(k, value)
                #print(k, s[count])
                assert_equal(key, k, "Unexpected key.")
                assert_equal(value, s[count], "Unexpected content.")
                
                count += 1
        
        assert_equal(count, 2, "Unexpected number of samples.")
        
class TestScalarsToLMDB:

    PREFIX = '\x08\x01\x10\x01\x18\x01(\x005\x00\x00'
    STR_MAPPINGS = {-2 : '\x00\xc0',
                    -1 : '\x80\xbf',
                     0 : '\x00\x00',
                     1 : '\x80?',
                     2 : '\x00@',
                     3 : '@@',
                     4 : '\x80@',
                     5 : '\xa0@',
                     6 : '\xc0@',
                     7 : '\xe0@',
                     8 : '\x00A',
                     9 : '\x10A',
                    10 : ' A'}
        
    @classmethod
    def setup_class(self):
        
        self.dir_tmp = tempfile.mkdtemp()
        
    @classmethod
    def teardown_class(self):
        
        shutil.rmtree(self.dir_tmp)   
            
#     def test_scalars_strX(self):
#         
#         # expected serialization of the test image
#         x = np.random.randint(-2, 11, size=(10, 1)) # [low, high)
#         
#         # use the module and test it
#         path_lmdb = os.path.join(self.dir_tmp, 'x2_lmdb')
#         tol.scalars_to_lmdb(x, path_lmdb)
#         assert_true(os.path.isdir(path_lmdb), "failed to save LMDB")
#         
#         env_src = lmdb.open(path_lmdb, readonly=True)
#         
#         c = 0
#         with env_src.begin() as txn:
#             for key, value in txn.cursor():
#                 #print(k, x[c], value)
#                 assert_equal(key, tol.IDX_FMT.format(c), "Unexpected key.")
#                 assert_equal(value,
#                              self.PREFIX + self.STR_MAPPINGS[x.ravel()[c]],
#                              "Unexpected content.")
#                 c += 1
#         
#         assert_equal(c, x.size, "Unexpected number of samples.")
        
    @patch('to_lmdb.caffe')
    @patch('to_lmdb.caffe.proto.caffe_pb2.Datum')
    def test_scalars_str(self, mock_dat, mock_caffe):
        
        # expected serialization of the test image
        x = np.random.randint(-2, 11, size=(10, 1)) # [low, high)
        ser_vals = [self.PREFIX + self.STR_MAPPINGS[v] for v in x.ravel()]
        
        # mock caffe calls made by our module
        mock_dat.return_value.SerializeToString = MagicMock(side_effect=ser_vals)
        mock_caffe.io.array_to_datum.return_value = caffe.proto.caffe_pb2.Datum()
        
        # use the module and test it
        path_lmdb = os.path.join(self.dir_tmp, 'x2_lmdb')
        tol.scalars_to_lmdb(x, path_lmdb)
        assert_true(os.path.isdir(path_lmdb), "failed to save LMDB")
        
        c = 0
        with lmdb.open(path_lmdb, readonly=True).begin() as txn:
            for key, value in txn.cursor():
                #print(k, x[c], value)
                assert_equal(key, tol.IDX_FMT.format(c), "Unexpected key.")
                assert_equal(value, ser_vals[c], "Unexpected content.")
                c += 1
        
        assert_equal(c, x.size, "Unexpected number of samples.")
        
    @patch('to_lmdb.caffe')
    @patch('to_lmdb.caffe.proto.caffe_pb2.Datum')
    def test_scalars_lut(self, mock_dat, mock_caffe):
        
        # expected serialization of the test image
        x = np.random.randint(-1, 4, size=(10, 1)) # [low, high)
        
        def lut(value):
            return value-1
        
        ser_vals = [self.PREFIX + self.STR_MAPPINGS[v-1] for v in x.ravel()]
        
        # mock caffe calls made by our module
        mock_dat.return_value.SerializeToString = MagicMock(side_effect=ser_vals)
        mock_caffe.io.array_to_datum.return_value = caffe.proto.caffe_pb2.Datum()
        
        # use the module and test it
        path_lmdb = os.path.join(self.dir_tmp, 'xlut_lmdb')
        tol.scalars_to_lmdb(x, path_lmdb, lut=lut)
        assert_true(os.path.isdir(path_lmdb), "failed to save LMDB")
        
        c = 0
        with lmdb.open(path_lmdb, readonly=True).begin() as txn:
            for key, value in txn.cursor():
                #print(k, x[c], value)
                assert_equal(key, tol.IDX_FMT.format(c), "Unexpected key.")
                assert_equal(value, ser_vals[c], "Unexpected content.")
                c += 1
        
        assert_equal(c, x.size, "Unexpected number of samples.")
        
    @patch('to_lmdb.caffe')
    @patch('to_lmdb.caffe.proto.caffe_pb2.Datum')
    def test_scalars_str_list_of_one(self, mock_dat, mock_caffe):
        
        # expected serialization of the test image
        x = np.random.randint(-2, 11) # [low, high), single integer
        
        # mock caffe calls made by our module
        v = self.PREFIX + self.STR_MAPPINGS[x]
        mock_dat.return_value.SerializeToString.return_value = self.PREFIX + self.STR_MAPPINGS[x]
        mock_caffe.io.array_to_datum.return_value = caffe.proto.caffe_pb2.Datum()
        
        # use the module and test it
        path_lmdb = os.path.join(self.dir_tmp, 'test_scalars_str_single_lmdb')
        tol.scalars_to_lmdb([x], path_lmdb)
        assert_true(os.path.isdir(path_lmdb), "failed to save LMDB")
        
        c = 0
        with lmdb.open(path_lmdb, readonly=True).begin() as txn:
            for key, value in txn.cursor():
                #print(k, x[c], value)
                assert_equal(key, tol.IDX_FMT.format(c), "Unexpected key.")
                assert_equal(value, self.PREFIX + self.STR_MAPPINGS[x],
                             "Unexpected content.")
                c += 1
        
        assert_equal(c, 1, "Unexpected number of samples.")
        
    @patch('to_lmdb.caffe')
    @patch('to_lmdb.caffe.proto.caffe_pb2.Datum')
    def test_scalars_str_single_int(self, mock_dat, mock_caffe):
        
        # expected serialization of the test image
        x = np.random.randint(-2, 11) # [low, high), single integer
        
        # mock caffe calls made by our module
        v = self.PREFIX + self.STR_MAPPINGS[x]
        mock_dat.return_value.SerializeToString.return_value = self.PREFIX + self.STR_MAPPINGS[x]
        mock_caffe.io.array_to_datum.return_value = caffe.proto.caffe_pb2.Datum()
        
        # use the module and test it
        path_lmdb = os.path.join(self.dir_tmp, 'test_scalars_str_single_lmdb')
        tol.scalars_to_lmdb(x, path_lmdb)
        assert_true(os.path.isdir(path_lmdb), "failed to save LMDB")
        
        c = 0
        with lmdb.open(path_lmdb, readonly=True).begin() as txn:
            for key, value in txn.cursor():
                #print(k, x[c], value)
                assert_equal(key, tol.IDX_FMT.format(c), "Unexpected key.")
                assert_equal(value, self.PREFIX + self.STR_MAPPINGS[x],
                             "Unexpected content.")
                c += 1
        
        assert_equal(c, 1, "Unexpected number of samples.")
        
    def test_scalars_invalid_scalars(self):
        
        assert_raises(AttributeError,
              tol.scalars_to_lmdb,
              [np.random.randint(-2, 11, size=(2, 3))],
              os.path.join(self.dir_tmp, 'xx_lmdb'))
        
        assert_raises(AttributeError,
                      tol.scalars_to_lmdb,
                      [np.random.randint(-2, 11, size=(2, 3, 4))],
                      os.path.join(self.dir_tmp, 'xx_lmdb'))
        
        