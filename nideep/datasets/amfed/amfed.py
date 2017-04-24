'''
Created on Mar 28, 2017

@author: kashefy
'''
import os
import logging
from collections import namedtuple
import nideep.iow.file_system_utils as fs
from comparables import list_comparables
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import cPickle as pickle
from preprocessing import crop_face


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
    INDEX_FILE = 'amfed-index.csv'

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

    def as_dataframe(self):
        index_path = self.cache_dir + AMFED.INDEX_FILE
        if os.path.isfile(index_path):
            result_df = pd.read_csv(index_path)
        else:
            entities = self.find_data_intersection()
            count = 0
            result = []
            for _, e in entities.iterrows():
                entity_records = self.load_file(e)
                result += entity_records
                print 'Finished video %d/%d' % (count + 1, len(entities))
                count += 1

            result_df = pd.DataFrame(result)
            result_df.to_csv(index_path)

        return result_df

    def as_numpy_array(self):
        df = self.as_dataframe()
        df = df[df.valid]
        # X_train = np.array([cv2.imread(e, cv2.IMREAD_GRAYSCALE) for e in df['features'].tolist()])
        X_train = np.array([cv2.imread(e) for e in df['features'].tolist()])
        y_train = np.array([pickle.load(open(e, 'rb')) for e in df['labels'].tolist()])
        return X_train, y_train

    def load_file(self, entity):
        """
        Loads a specific video of the dataset, preprocess it (frame extraction, resizing) and return it as a numpy array
        :return: A list of dictionaries
        """
        labels_df = pd.read_csv(entity.au_label)
        landmarks_df = pd.read_csv(entity.landmarks, header=None)

        vidcap = cv2.VideoCapture(entity.video)
        success, image = vidcap.read()
        count = 0
        success = True
        result = []

        while success:

            label_file = self.__labels_filename(entity, count)
            frame_file = self.__frame_filename(entity, count)

            record = {'features': frame_file, 'labels': label_file, 'message': '', 'valid': True}

            if labels_df.empty:
                record['valid'] = False
                record['message'] = 'Empty labels file'
            else:
                if not (os.path.isfile(label_file) and os.path.isfile(frame_file)):
                    success, msg = self.preprocess_frame(entity, image, count, labels_df=labels_df,
                                                         landmarks_df=landmarks_df)
                    record['valid'] = success
                    record['message'] = msg

            result.append(record)
            success, image = vidcap.read()
            count += 1

        return result

    def preprocess_frame(self, entity, frame, frame_number, labels_df=None, landmarks_df=None):
        label_file = self.__labels_filename(entity, frame_number)
        frame_file = self.__frame_filename(entity, frame_number)

        time = AMFED.__frame_to_timestamp(frame_number)

        if labels_df is None:
            labels_df = pd.read_csv(entity.au_label)

        if not landmarks_df is None:
            landmarks_df = pd.read_csv(entity.landmarks, header=None)

        labels = []
        aucs = ['Smile', 'AU02', 'AU04', 'Trackerfail']  # TODO remove hardcoded aucs

        label_columns = [e for e in aucs if e in list(labels_df)]
        if label_columns:
            label_row = labels_df[labels_df['Time'] <= time].iloc[-1].loc[label_columns]
            labels = [1 if e >= 50 else 0 for e in label_row]

        if frame_number < landmarks_df.shape[0]:
            left_eye = (
                landmarks_df.iloc[[frame_number]][0][frame_number], landmarks_df.iloc[[frame_number]][1][frame_number])
            right_eye = (
                landmarks_df.iloc[[frame_number]][2][frame_number], landmarks_df.iloc[[frame_number]][3][frame_number])
        else:
            return False, 'Frame number > number of landmark file rows'

        if not all(v == 0 for v in (left_eye + right_eye)) and len(labels) == len(aucs):
            # print 'Read a new frame from video: ', count

            if labels[-1]:
                return False, 'Tracker fail: human labeled failure'

            cropped = np.array(crop_face(Image.fromarray(frame), eye_left=left_eye, eye_right=right_eye,
                                         offset_pct=(0.25, 0.25), dest_sz=(64, 64)))

            # gray_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

            pickle.dump(labels[:-1], open(label_file, 'wb'))
            cv2.imwrite(frame_file, cropped)  # save frame as JPEG file
            return True, 'Success'
        else:
            return False, 'Tracker fail: no landmarks available'

    def __labels_filename(self, entity, frame_number):
        video_name = entity.video.split('/')[-1][:-4]
        return self.cache_dir + video_name + '-labels%d.p' % frame_number

    def __frame_filename(self, entity, frame_number):
        video_name = entity.video.split('/')[-1][:-4]
        return self.cache_dir + video_name + '-frame%d.jpg' % frame_number

    @staticmethod
    def __timestamp_to_frame(time):
        return time / (1.0 / 14)

    @staticmethod
    def __frame_to_timestamp(frame):
        return frame * (1.0 / 14)

    def set_subset(self, entities):
        self.entities = entities
        
    def dense_labels(self):
        raise NotImplementedError # TODO: Implement

    def __init__(self, dir_prefix, video_type=VIDEO_TYPE_FLV, cache_dir=None):
        '''
        Constructor
        '''
        self.logger = logging.getLogger(__name__)
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
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
    amfed = AMFED(dir_prefix='/mnt/raid/data/ni/dnn/AMFED/', video_type=AMFED.VIDEO_TYPE_AVI,
                  cache_dir='/mnt/raid/data/ni/dnn/rparra/cache/')
    amfed.as_numpy_array()
