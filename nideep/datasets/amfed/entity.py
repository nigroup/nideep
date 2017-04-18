import os
import logging
import numpy as np
import cv2

class Entity(object):
    
    def timestamps_sec(self):
         """ Get timestamp of each frame from video file """
        cap = cv2.VideoCapture(self.path_video)
        ret = True
        t = []
        while cap.isOpened() and ret:
            # Capture frame-by-frames
            t.append( cap.get( cv2.CAP_PROP_POS_MSEC ) / 1000. )
            ret = cap.grab()
        # When everything done, release the capture
        cap.release()
        return t
    
    def dense_labels_dict(self):
         """ Load dense labels for each column in au labels file into dictionary"""
        labels, col_names = self.dense_labels()
        labels_dict = {}
        for idx, c in enumerate(col_names):
            labels_dict[c] = labels[:, idx]
        return labels_dict
    
    def dense_labels(self):
         """ Get dense labels for all columns in au labels """
        onsets = np.genfromtxt(self.path_au_labels, dtype=float, delimiter=',', names=True)
        t_dense = self.timestamps_sec()
        cols = list(onsets.dtype.names[1:]) # omit Time from header
        labels = np.zeros((len(t_dense), len(cols)), dtype='float')
        if onsets.shape and len(onsets) > 1: 
            for cur, next in zip(onsets[:-1], onsets[1:]):
                t_on = cur[0]
                t_off = next[0]
                valid_rows = np.logical_and(t_dense >= t_on, t_dense < t_off)
                labels[valid_rows, :] = list(cur)[1:]
            labels[t_dense >= t_off, :] = list(next)[1:] # fill remaining rows
        return labels, cols
    
    def __init__(self, path_video, path_au_labels, path_landmarks):
        '''
        Constructor
        '''
        self.logger = logging.getLogger(__name__)
        self.path_video = path_video
        if not os.path.isfile(self.path_video):
            self.logger.warning("Video file does not exist (%s)" % self.path_video)
        self.path_au_labels = path_au_labels
        if not os.path.isfile(self.path_au_labels):
            self.logger.warning("AU Labels file does not exist (%s)" % self.path_au_labels)
        self.path_landmarks = path_landmarks
        if not os.path.isfile(self.path_landmarks):
            self.logger.warning("Landmarks file does not exist (%s)" % self.path_landmarks)
        