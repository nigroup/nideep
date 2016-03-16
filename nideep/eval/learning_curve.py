'''
Created on Mar 1, 2016

@author: kashefy
'''
import os
import numpy as np       
import parse_log as pl
from eval_utils import Phase
import nideep.iow.file_system_utils as fs
from log_utils import is_caffe_info_log
    
def cleanup_caffe_logname(fname):
    
    fname = fname.replace('log', '')
    fname = fname.replace('caffe', '')
    fname = fname.replace('.', '')
    return fname

class LearningCurve(object):
    '''
    classdocs
    '''
    def list(self, key, phase=Phase.TEST):
                
        if phase.lower() == Phase.TEST.lower():
            
            if 'loss' in key.lower() and 'loss' in self.test_keys:
                key = 'loss'
            elif 'accuracy' in key.lower() and 'accuracy' in self.test_keys:
                key = 'accuracy'
                
            return np.array([[x[key]] for x in self.test_dict_list])
        else:
            return np.array([[x[key]] for x in self.train_dict_list])
        
    def name(self):
    
        name = os.path.basename(self.path_log)
        name, _ = os.path.splitext(name)
        return cleanup_caffe_logname(name)
    
    def parse(self):
        
        log_data = pl.parse_log(self.path_log)
        # allow for backwards compatibility
        if len(log_data) == 4:
            self.train_dict_list, self.train_keys, self.test_dict_list, self.test_keys = log_data
        else:
            self.train_dict_list, self.test_dict_list = log_data
            if len(self.train_dict_list) > 0:
                self.train_keys = self.train_dict_list[0].keys()
            else:
                self.train_keys = []
                
            if len(self.test_dict_list) > 0:
                self.test_keys = self.test_dict_list[0].keys()
            else:
                self.test_keys = []

        return self.train_keys, self.test_keys

    def __init__(self, path_log):
        '''
        Constructor
        '''
        self.path_log = path_log
        
def LearningCurveFromPath(p):
    
    if os.path.isfile(p):
        return LearningCurve(p)
    elif os.path.isdir(p):
        log_paths = fs.gen_paths(p, func_filter=is_caffe_info_log)
        if len(log_paths) > 0:
            return LearningCurve(log_paths[-1])
        else:
            return None
    else:
        raise IOError("%s: No such file or directory" % (p,))
    