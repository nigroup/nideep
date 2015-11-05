'''
Created on Nov 5, 2015

A module to mock caffe structures for testing

@author: kashefy
'''

class proto:
    class caffe_pb2:
        class Datum:
            def SerializeToString(self):
                return "mock_this"
            
class io:
    @staticmethod
    def array_to_datum(s):
        return proto.caffe_pb2.Datum()