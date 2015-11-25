'''
Created on Oct 30, 2015

@author: kashefy
'''
from nose.tools import assert_equal, assert_almost_equals, assert_list_equal
from mock import patch, PropertyMock
import os
import tempfile
import shutil
import lmdb
import caffe
import read_lmdb as r

class TestReadLMDB:

    @classmethod
    def setup_class(self):
        
        self.dir_tmp = tempfile.mkdtemp()
        self.path_lmdb = os.path.join(self.dir_tmp, 'x_lmdb')
        
        # write fake data to lmdb
        db = lmdb.open(self.path_lmdb, map_size=int(1e12))
        with db.begin(write=True) as in_txn:
            for idx in range(5):
                in_txn.put('{:0>10d}'.format(idx), "%s" % (idx * 10))
        db.close()
        
    @classmethod
    def teardown_class(self):
        
        shutil.rmtree(self.dir_tmp)
        
    @patch('read_lmdb.caffe.proto.caffe_pb2.Datum')
    def test_read_labels(self, mock_dat):
        
        # mock methods and properties of Datum objects
        mock_dat.return_value.ParseFromString.return_value = ""
        type(mock_dat.return_value).label = PropertyMock(side_effect=range(5))
        
        assert_list_equal(r.read_labels(self.path_lmdb), range(5))
        
    