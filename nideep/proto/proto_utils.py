'''
Created on Sep 21, 2015

@author: kashefy
'''
from google.protobuf import text_format
from caffe.proto import caffe_pb2

def copy_msg(src, typ):

    dst = typ()
    dst.CopyFrom(src)
    return dst

def copy_net_params(src):

    return copy_msg(src, caffe_pb2.NetParameter)

class Parser(object):
    '''
    classdocs
    '''
    def from_net_params_file(self, fpath):

        config = caffe_pb2.NetParameter()
        return self.from_file(fpath, config)

    def from_file(self, fpath, dst_obj):

        with open(fpath, "r") as f:
            text_format.Merge(str(f.read()), dst_obj)
        return dst_obj

    def __init__(self):
        '''
        Constructor
        '''

