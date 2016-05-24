# NI Deep - Deep Learning Tools from the NI Group at TU-Berlin

[![Build Status](https://travis-ci.org/nigroup/nideep.svg?branch=master)](https://travis-ci.org/nigroup/nideep)
[![Coverage Status](https://coveralls.io/repos/github/nigroup/nideep/badge.svg?branch=master)](https://coveralls.io/github/nigroup/nideep?branch=master)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

This is a collection of scripts, utilities and examples to use with [caffe](http://caffe.berkeleyvision.org/). These are either too use case specific and are therefore kept outside of the Caffe framework.
They may also be considered completely independent and potentially useful in combination with other frameworks.

Use cases include:
* i/o of custom data to/from lmdb
* generating lmdb for select benchmark datasets (adding support for more is welcome)
* parsing caffe logs (incomplete, wraps around existing parser from Caffe)
* editing caffe protobuf messages for network definitions

# Disclaimer:
Some automated testing is in place. However, do not assume the functionality is free of bugs. Feel free to inspect them, scrutinize them. Feedback on making them more usable for others is welcome.

Beware: The documentation is pretty scarce. We're working on it.

# Getting Started:

## Dependencies:
* Caffe with python support (pycaffe)
* LMDB (apt-get and pip install)
* cv2 (you probably already have this if you've built caffe)
* PIL (pip install)
* h5py

# Examples:

## PASCAL-Context to LMDB

Download val_59.txt from [here](https://gist.github.com/kashefy/78415dd397accb47872a/raw/761b280d6de022958f8f8c9bc64fa56432124cb2/val_59.txt).

Download 59_labels.txt from [here](https://gist.githubusercontent.com/kashefy/b598193e03d21432f327/raw/036272de91604991dd1daeccb69fa0cfc6611c1d/59_labels.txt).

    from nideep.datasets.pascal_context_to_lmdb import pascal_context_to_lmdb
    val_list_path = os.path.expanduser('~/data/PASCAL-Context/val_59.txt')
    with open(val_list_path, 'r') as f:
        val_list = f.readlines()
        val_list = [l.translate(None, ''.join('\n')) for l in val_list if len(l) > 0]
    
    nt, nv, fpath_imgs_train, fpath_labels_train, fpath_imgs_val, fpath_labels_val = \
    pascal_context_to_lmdb(os.path.expanduser('~/data/VOCdevkit/VOC2012/JPEGImages'),
                           os.path.expanduser('~/data/PASCAL-Context/trainval'),
                           os.path.expanduser('~/data/PASCAL-Context/labels.txt'),
                           os.path.expanduser('~/data/PASCAL-Context/59_labels.txt'),
                           '',
                           os.path.expanduser('~/data/PASCAL-Context/'),
                           val_list=val_list
                           )
    
    print "size: %d" % nt, nv, fpath_imgs_train, fpath_labels_train, fpath_imgs_val, fpath_labels_val
