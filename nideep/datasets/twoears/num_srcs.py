'''
Created on May 10, 2017

@author: kashefy
'''
import numpy as np
import h5py
from sklearn.metrics import confusion_matrix
from summary import get_model_keys

class NumSrcs(object):
    '''
    classdocs
    '''
    @property
    def protected_value(self):
        return self._class_names

    @protected_value.setter
    def protected_value(self, class_names):
        self._class_names = class_names
        
    def micro(self, gt, pred):
        c = confusion_matrix(gt, pred)
        tp = np.sum(c[cl, cl] for cl in self.class_names)
        tn = np.sum(c[cl, cl] for cl in self.class_names)
        
    def measure(self, fpath_infer, num_points, key_label='label_nsrcs'):

        with h5py.File(fpath_infer, 'r') as h:
            if num_points is None:
                num_points = len(h[key_label])
            model_stats = get_model_keys(h, key_label=key_label)
    
            for m in model_stats:
                key_pred = m['pred_name']
                gt = h[key_label]
                pred = h[key_pred]
                if m['pred_type'] == 'nsrcs':
                    gt = np.squeeze(gt[:num_points]).ravel()
                    pred = np.argmax(pred[:num_points], axis=1).ravel()
                    m['%s_conf' % m['pred_type']] = eval_nsrcs_confusion(gt, pred, classnames=classnames)
        return model_stats

    def __init__(self, params):
        '''
        Constructor
        '''
        self._class_names = range(5)
        
if __name__ == '__main__':
    from nideep.datasets.twoears.summary import load_obj
    c = load_obj('/home/kashefy/Downloads/xG_brir_conv_te_multi_xG_brir_c_03_2_02_2_900K_iter_492180_confusion_nsrcs')
    
    pass
        