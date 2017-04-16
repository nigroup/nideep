'''
Created on Mar 28, 2017

@author: kashefy
'''
import os
import logging
from collections import namedtuple
import nideep.iow.file_system_utils as fs
from comparables import list_comparables


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
        self.logger.log([logging.WARNING, logging.DEBUG][len(p)>0],
                        "%d entities found under %s" % (len(p), p))
        
    def from_list_to_typle(self, items, lists, list_names):
        
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
        common_entities = self.from_list_to_typle(common_entity_names, lists, list_names)
        self.entities = common_entities
        return common_entities
    
    def set_subset(self, entities):
        self.entities = entities
        
    def dense_labels(self):
        raise NotImplementedError # TODO: Implement

    def __init__(self, dir_prefix, video_type=DIRNAME_FLV):
        '''
        Constructor
        '''
        self.logger = logging.getLogger(__name__)
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
        self.entities = []
            