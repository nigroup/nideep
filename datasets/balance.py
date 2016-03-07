import numpy as np

CLNAME_OTHER = 'other'

def balance_class_count(self, feats, labls, other_clname=CLNAME_OTHER):
    
    bal = Balancer(labls)
    class_count = bal.get_class_count(other_clname=other_clname)
    idxs = bal.get_idxs_to_balance_class_count(class_count.values())
    idxs = np.random.shuffle(idxs) # this only shuffles the array along the first index of a multi-dimensional array
    return feats[idxs], labls[idxs]

class Balancer(object):
    '''
    Balance class counts
    '''
    def get_class_count(self, other_clname=CLNAME_OTHER):
        
        self.has_other_cl = other_clname is not None and other_clname != ''
        
        if self.l.ndim == 2:
            class_count = np.sum(self.l, axis=0)
            other_count = int(len(self.l) - np.sum(class_count))
        d = {}
        for i, x in enumerate(class_count):
            d[i] = int(x)
        if self.has_other_cl:
            d[other_clname] = other_count
        return d
    
    def get_idxs_to_balance_class_count(self, class_counts):
    
        mx = np.max(class_counts)
        num_examples, num_classes = self.l.shape
        idxs = np.arange(num_examples).reshape(-1, 1)
        for i, c in enumerate(class_counts):
            delta_ = mx - c
            if delta_ > 0:
                if i < num_classes:
                    rows = np.where(self.l[:, i]==1)[0]
                else: # cannot index other class
                    rows = np.where(np.sum(self.l, axis=-1) <= 0)[0]
                rows_to_sample = rows[np.random.randint(0, high=rows.size,
                                                        size=(delta_, 1))
                                      ]
                idxs = np.vstack((idxs, rows_to_sample))
        return idxs

    def __init__(self, labls):
        '''
        Constructor
        labls -- ground truth
        '''
        self.l = labls
        self.has_other_cl = True
        