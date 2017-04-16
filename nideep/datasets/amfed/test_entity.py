'''
Created on Oct 28, 2015

@author: kashefy
'''
import os
import tempfile
import random
import shutil
import string
from nose.tools import assert_equals, assert_true, \
    assert_list_equal
from mock import patch, MagicMock
from entity import Entity
import numpy as np

def generate_random_entity_name():
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(9))

def write_dummy_txt(p):
    with open(p, 'wb') as h:
        h.write("foo")
        
class TestEntity:
    
    NUM_FRAMES = 5
    
    @classmethod
    def setup_class(self):

        self.dir_tmp = tempfile.mkdtemp()
        self.entity_name = generate_random_entity_name()
        self.path_video = os.path.join(self.dir_tmp, self.entity_name + '.flv')
        self.path_au_labels = os.path.join(self.dir_tmp, self.entity_name + '-label.csv')
        self.path_landmarks = os.path.join(self.dir_tmp, self.entity_name + '-landmarks.txt')
        
        # fake video files
        write_dummy_txt(os.path.join(self.path_video))
        write_dummy_txt(os.path.join(self.path_au_labels))
        write_dummy_txt(os.path.join(self.path_landmarks))
                        
    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.dir_tmp)
        
    @patch('cv2.VideoCapture')
    def test_timesteps_sec(self, mock_cap):

        node_instance = mock_cap.return_value
        pos_sec = np.sort(np.random.rand(TestEntity.NUM_FRAMES)*1000).tolist()
        node_instance.get = MagicMock(side_effect=pos_sec)
        node_instance.isOpened = \
            MagicMock(side_effect=[True]*(TestEntity.NUM_FRAMES) + [False])
        node_instance.grab.return_value = True
        
        e = Entity(self.path_video, self.path_au_labels, self.path_landmarks)
        t = e.timestamps_sec()
        assert_list_equal(t, [x/1000. for x in pos_sec])

#    def test_intersection(self):
##        e = Entity('/home/kashefy/Downloads/AMFEDx/AMFED/Video - AVI/0d48e11a-2f87-4626-9c30-46a2e54ce58e.avi',
#        e = Entity('/home/kashefy/Downloads/AMFEDx/AMFED/Videos - FLV/0d48e11a-2f87-4626-9c30-46a2e54ce58e.flv',
#                   '/home/kashefy/Downloads/AMFEDx/AMFED/AU Labels/0d48e11a-2f87-4626-9c30-46a2e54ce58e-label.csv',
#                   '/home/kashefy/Downloads/AMFEDx/AMFED/Landmark Points/0d48e11a-2f87-4626-9c30-46a2e54ce58e-landmarks.txt')
#        t = e.timestamps_sec()
#        e.dense_labels()
#        e = Entity('/home/kashefy/Downloads/AMFEDx/AMFED/Video - AVI/0d48e11a-2f87-4626-9c30-46a2e54ce58e.avi',
##        e = Entity('/home/kashefy/Downloads/AMFEDx/AMFED/Videos - FLV/0d48e11a-2f87-4626-9c30-46a2e54ce58e.flv',
#                   '/home/kashefy/Downloads/AMFEDx/AMFED/AU Labels/0d48e11a-2f87-4626-9c30-46a2e54ce58e-label.csv',
#                   '/home/kashefy/Downloads/AMFEDx/AMFED/Landmark Points/0d48e11a-2f87-4626-9c30-46a2e54ce58e-landmarks.txt')
#       
#        t2 = e.timestamps_sec()
#        
#        for x,y, in zip(t,t2):
#            print x,y, x==y
##        t = np.transpose(np.reshape(t, (1,-1)))
##        labels_dense = e.dense_labels()
##        labels_dense = np.hstack((t, labels_dense))
##        print labels_dense.shape
##        np.savetxt('/home/kashefy/Downloads/AMFEDx/AMFED/Landmark Points/x.txt',
##                      labels_dense, delimiter=',')