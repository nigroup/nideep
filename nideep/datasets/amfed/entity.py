import os
import logging
import numpy as np
import cv2

class Entity(object):
    
    def timestamps_sec(self):
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
    
    def dense_labels(self):
        
        #header = np.loadtxt(self.path_au_labels, delimiter=',', skiprows=1)
        #labels = np.loadtxt(self.path_au_labels, delimiter=',', skiprows=1)
        onsets = np.genfromtxt(self.path_au_labels, dtype=float, delimiter=',', names=True)
        print onsets.shape
        print onsets.dtype.names
        #print onsets
        t_dense = self.timestamps_sec()
        cols = onsets.dtype.names[1:] # omit Time from header
        labels = np.zeros((len(t_dense), len(cols)), dtype='float')
#        t_onsets = [l[0] for l in onsets]
#        while onset_idx < len(t_onsets)
#        for t in t_dense:
        if len(onsets) > 1: 
            for cur, next in zip(onsets[:-1], onsets[1:]):
                t_on = cur[0]
                t_off = next[0]
                valid_rows = np.logical_and(t_dense >= t_on, t_dense < t_off)
                labels[valid_rows, :] = list(cur)[1:]
        return labels
    
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
        