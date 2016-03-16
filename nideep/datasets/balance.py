import numpy as np

CLNAME_OTHER = 'other'

class Balancer(object):
    '''
    Balance class counts
    '''
    def get_class_count(self, other_clname=CLNAME_OTHER):
        """
        Return per-class occurrence count
        
        Keyword argumented:
        other_clname -- a name for an overall negative class (all inactive)
        """
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
        """
        Determine indices with with which we can sample from the dataset
        and get a balanced inter-class distribution
        """
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
        return idxs.ravel()

    def __init__(self, labls):
        '''
        Constructor
        labls -- ground truth
        '''
        self.l = labls
        self.has_other_cl = True
        