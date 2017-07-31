'''
Created on Mar 28, 2017

@author: kashefy
'''
import logging
import os
import random
from collections import namedtuple

import numpy as np
import pandas as pd

import nideep.iow.file_system_utils as fs
from comparables import list_comparables
from nideep.datasets.amfed.entity import Entity


class AMFED(object):
    '''
    classdocs
    '''
    DIRNAME_AU_LABELS = 'AU Labels'
    DIRNAME_LANDMARKS = 'Landmark Points'
    VIDEO_TYPE_AVI = 'AVI'
    VIDEO_TYPE_FLV = 'FLV'
    DIRNAME_AVI = 'Video - ' + VIDEO_TYPE_AVI
    DIRNAME_FLV = 'Videos - ' + VIDEO_TYPE_FLV
    SUFFIX_AU_LABELS = '-label'
    SUFFIX_LANDMARKS = '-landmarks'
    KEY_VIDEO = 'video'
    KEY_AU_LABELS = 'au_label'
    KEY_LANDMARKS = 'landmarks'
    TMP_TRAIN_IMAGES = 'xtrain.dat'
    TMP_TRAIN_LABELS = 'ytrain.dat'
    TMP_TEST_IMAGES = 'xtest.dat'
    TMP_TEST_LABELS = 'ytest.dat'

    @staticmethod
    def is_video(p):
        return os.path.splitext(p)[-1] in ['.avi', '.flv']

    @staticmethod
    def is_au_label(p):
        return p.endswith('-label.csv')

    @staticmethod
    def is_landmarks(p):
        return os.path.splitext(p)[0].endswith('-landmarks')

    def get_paths_videos(self):

        p = fs.gen_paths(self.dir_videos, AMFED.is_video)
        self.log_num_paths(p)
        return p

    def get_paths_au_labels(self):

        p = fs.gen_paths(self.dir_au_labels, AMFED.is_au_label)
        self.log_num_paths(p)
        return p

    def get_paths_landmarks(self):

        p = fs.gen_paths(self.dir_landmarks, AMFED.is_landmarks)
        self.log_num_paths(p)
        return p

    def log_num_paths(self, p):
        self.logger.log([logging.WARNING, logging.DEBUG][len(p) > 0],
                        "%d entities found under %s" % (len(p), p))

    def from_list_to_tuple(self, items, lists, list_names):

        AMFEDEntity = namedtuple('AMFEDEntity', list_names)
        intersection = []
        for e in items:
            for l, lname in zip(lists, list_names):
                if lname == AMFED.KEY_VIDEO:
                    path_video = l[l.index(e)].p
                elif lname == AMFED.KEY_AU_LABELS:
                    path_au_labels = l[l.index(e)].p
                elif lname == AMFED.KEY_LANDMARKS:
                    path_landmarks = l[l.index(e)].p
                else:
                    self.logger.warning("Unknown list name %s", lname)
            intersection.append(AMFEDEntity(path_video,
                                            path_au_labels,
                                            path_landmarks))
        return intersection

    def find_data_intersection(self):

        paths_videos = self.get_paths_videos()
        paths_au_labels = self.get_paths_au_labels()
        paths_landmarks = self.get_paths_landmarks()

        lists = [list_comparables(paths_videos)]
        list_names = [AMFED.KEY_VIDEO]
        if paths_au_labels is not None and len(paths_au_labels) > 0:
            lists.append(list_comparables(paths_au_labels, suffix='-label'))
            list_names.append(AMFED.KEY_AU_LABELS)
        if paths_au_labels is not None and len(paths_landmarks) > 0:
            lists.append(list_comparables(paths_landmarks, suffix='-landmarks'))
            list_names.append(AMFED.KEY_LANDMARKS)

        common_entity_names = list(set(lists[0]).intersection(*lists[1:]))
        common_entities = self.from_list_to_tuple(common_entity_names, lists, list_names)
        return pd.DataFrame([c._asdict() for c in common_entities])

    def as_numpy_array(self, projection=['Smile', 'Trackerfail'], valid_only=True, train_proportion=0.8):
        tmp_train_images = os.path.join(self.cache_dir, self.TMP_TRAIN_IMAGES)
        tmp_train_labels = os.path.join(self.cache_dir, self.TMP_TRAIN_LABELS)
        tmp_test_images = os.path.join(self.cache_dir, self.TMP_TEST_IMAGES)
        tmp_test_labels = os.path.join(self.cache_dir, self.TMP_TEST_LABELS)
        if all([os.path.exists(f) for f in [tmp_train_images, tmp_train_labels, tmp_test_images, tmp_test_labels]]):
            X_train_memmap = np.memmap(tmp_train_images).reshape((-1, 64, 64, self.c_dim))
            y_train_memmap = np.memmap(tmp_train_labels).reshape((-1, len(projection) - 1))
            X_test_memmap = np.memmap(tmp_test_images).reshape((-1, 64, 64, self.c_dim))
            y_test_memmap = np.memmap(tmp_test_labels).reshape((-1, len(projection) - 1))
        else:
            entities = self.find_data_intersection()
            test_size = int(entities.shape[0] * (1 - train_proportion))
            test_idxs = random.sample(range(entities.shape[0]), test_size)
            count = 0
            train = []
            test = []
            for i, e in entities.iterrows():
                entity = Entity(e.video, e.au_label, e.landmarks, self.cache_dir)
                entity_records = entity.frames(projection=projection, valid_only=valid_only)
                if i in test_idxs:
                    test += entity_records
                else:
                    train += entity_records
                self.logger.info('Finished video %d/%d' % (count + 1, len(entities)))
                count += 1

            X_train_memmap, y_train_memmap = self.__list_to_array(tmp_train_images, tmp_train_labels, train)
            if test_idxs:
                X_test_memmap, y_test_memmap = self.__list_to_array(tmp_test_images, tmp_test_labels, test)

        p = np.random.permutation(X_train_memmap.shape[0])
        return X_train_memmap[p], y_train_memmap[p], X_test_memmap, y_test_memmap

    def __list_to_array(self, tmp_images_path, tmp_labels_path, result):
        # trim = lambda d: {k: v for k, v in d.iteritems() if k not in ['features']}
        # pickle.dump([trim(d) for d in result], open("result.p", "wb"))

        X = np.concatenate([np.expand_dims(r['features'], axis=0) for r in result])
        y = np.concatenate([np.expand_dims(r['labels'][:-1], axis=0) for r in result])

        X_memmap = np.memmap(tmp_images_path, shape=X.shape, mode='w+')
        y_memmap = np.memmap(tmp_labels_path, shape=y.shape, mode='w+')
        X_memmap[:] = X[:]
        y_memmap[:] = y[:]

        return X_memmap, y_memmap


    @staticmethod
    def __timestamp_to_frame(time):
        return time / (1.0 / 14)

    @staticmethod
    def __frame_to_timestamp(frame):
        return frame * (1.0 / 14)

    def set_subset(self, entities):
        self.entities = entities

    def __init__(self, dir_prefix, video_type=VIDEO_TYPE_FLV, cache_dir=None, grayscale=False):
        '''
        Constructor
        '''
        self.logger = logging.getLogger(__name__)
        self.cache_dir = cache_dir
        self.c_dim = 1 if grayscale else 3
        if cache_dir and not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
            self.logger.info("Creating cache directory at %s" % os.path.abspath(self.cache_dir))
        self.dir_au_labels = os.path.join(dir_prefix, AMFED.DIRNAME_AU_LABELS)
        if not os.path.isdir(self.dir_au_labels):
            self.logger.warning("AU Labels directory does not exist (%s)" % self.dir_au_labels)
        self.dir_landmarks = os.path.join(dir_prefix, AMFED.DIRNAME_LANDMARKS)
        if not os.path.isdir(self.dir_landmarks):
            self.logger.warning("Landmarks directory does not exist (%s)" % self.dir_landmarks)
        if video_type == AMFED.VIDEO_TYPE_AVI:
            self.dir_videos = os.path.join(dir_prefix, AMFED.DIRNAME_AVI)
            self.logger.debug("Use AVIs")
        else:
            self.dir_videos = os.path.join(dir_prefix, AMFED.DIRNAME_FLV)
            self.logger.debug("Use FLVs")
        if not os.path.isdir(self.dir_videos):
            self.logger.warning("Videos directory does not exist (%s)" % self.dir_videos)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    amfed = AMFED(dir_prefix='/mnt/raid/data/ni/dnn/AMFED/', video_type=AMFED.VIDEO_TYPE_AVI,
                  cache_dir='/mnt/raid/data/ni/dnn/rparra/cache/')
    amfed.as_numpy_array()
