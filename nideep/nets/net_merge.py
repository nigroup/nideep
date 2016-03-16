'''
Created on Dec 18, 2015

@author: kashefy
'''
from sets import Set
import numpy as np
from google.protobuf import text_format
from nideep.proto.proto_utils import Parser

def merge_indep_net_spec(net_specs, suffix_fmt='_nidx_%02d'):
    
    data_tops = [l.top for n in net_specs for l in n.layer if l.type.lower() == 'data']
    data_tops = Set([item for sublist in data_tops for item in sublist])
    
    for idx, n in enumerate(net_specs):
    
        suffix = suffix_fmt % idx
        throw_away = []
        for l in n.layer:
            if l.type.lower() != 'data':
                
                l.name += suffix
                
                if np.prod([p.lr_mult for p in l.param]) == 0:
                    print "LAYER WITH FIXED WEIGHTS. MAKE SHARED?"
                    
                for b in list(l.bottom):
                    l.bottom.remove(b)
                    if b not in data_tops:
                        l.bottom.append(unicode(b+suffix))
                    else:
                        l.bottom.append(unicode(b)) # preserve order of layer bottoms, label as bottom has to come last
                    
                for b in list(l.top):
                    l.top.remove(b)
                    if b not in data_tops:
                        l.top.append(unicode(b+suffix))
                    else:
                        l.top.append(unicode(b)) # preserve order of layer tops
                        
            elif idx > 0:
                throw_away.append(l)
                
        for l in throw_away:
            n.layer.remove(l) # Data layers of first net only
                
    proto_str = ''    
    for idx, n in enumerate(net_specs):
        
        s = text_format.MessageToString(n)
        if idx > 0 and s.startswith("name:"):
            _, s = s.split('\n', 1)
        proto_str += s
        
    return proto_str

if __name__ == '__main__':
    
    fpath_net_1 = '/home/kashefy/models/dark/mnist/tx/tx1.prototxt'
    fpath_net_2 = '/home/kashefy/models/dark/mnist/tx/tx2.prototxt'
    
    n1 = Parser().from_net_params_file(fpath_net_1)
    n2 = Parser().from_net_params_file(fpath_net_2)
    n_str = merge_indep_net_spec([n1, n2])
    
    fpath_dst = "/home/kashefy/models/dark/mnist/tx/m.prototxt"
    
    with open(fpath_dst, 'w') as f:
        f.write(n_str)
    
    import caffe
    n = caffe.Net("/home/kashefy/models/dark/mnist/tx/m.prototxt", caffe.TRAIN)
    n = caffe.Net("/home/kashefy/models/dark/mnist/tx/m.prototxt", caffe.TEST)
    
    pass