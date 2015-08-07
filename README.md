# My Caffe Sandbox
This is a random collection of scripts/utilities to use with caffe.
Do not assume they're thorouhgly tested but please feel free to inspect them, scrutinize them. Feedback on making them more usable for others are welcome.

Beware: the documentation is pretty scarce. I'm working on it.

# Dependencies
* Caffe with python support (pycaffe)
* LMDB (apt-get and pip install)
* cv2 (you probably already have this if you've build caffe)
* PIL (pip install)

## PASCAL-Context to LMDB

Download val_59.txt from [here](https://gist.github.com/kashefy/78415dd397accb47872a/raw/761b280d6de022958f8f8c9bc64fa56432124cb2/val_59.txt).

    from pascal_context_to_lmdb import pascal_context_to_lmdb
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
