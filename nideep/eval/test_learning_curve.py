'''
Created on Oct 1, 2015

@author: kashefy
'''
import os
import sys
from nose.tools import assert_equal, assert_greater, assert_true, \
    assert_list_equal, assert_false, assert_raises, \
    assert_is_instance, assert_is_not_none, assert_is_none
import numpy as np
from eval_utils import Phase
from learning_curve import LearningCurve
from learning_curve import LearningCurveFromPath

CURRENT_MODULE_PATH = os.path.abspath(sys.modules[__name__].__file__)
ROOT_PKG_PATH = os.path.dirname(CURRENT_MODULE_PATH)
TEST_DATA_DIRNAME = 'test_data'
TEST_LOG_FILENAME = 'caffe.hostname.username.log.INFO.20150917-163712.31405'

class TestLearningCurve:

    @classmethod
    def setup_class(self):

        self.fpath = os.path.join(os.path.dirname(ROOT_PKG_PATH),
                                  TEST_DATA_DIRNAME,
                                  TEST_LOG_FILENAME)
        assert_true(os.path.isfile(self.fpath),
                    "Cannot test without log file.")

    def test_keys_parsed(self):

        lc = LearningCurve(self.fpath)
        train_keys, test_keys = lc.parse()
        assert_list_equal(train_keys, ['NumIters', 'Seconds', 'LearningRate', 'loss'])
        assert_list_equal(test_keys, ['NumIters', 'Seconds', 'LearningRate', 'accuracy', 'loss'])

    def test_list(self):

        lc = LearningCurve(self.fpath)
        lc.parse()
        x = lc.list('NumIters')
        assert_greater(x.size, 0)
        loss = lc.list('loss')
        assert_equal(x.shape, loss.shape)
        acc = lc.list('accuracy')
        assert_equal(x.shape, acc.shape)

    def test_list_num_iters(self):

        lc = LearningCurve(self.fpath)
        lc.parse()
        x = lc.list('NumIters')
        dx = np.diff(x)
        assert_true(np.all(dx > 0))

    def test_list_loss_acc(self):

        lc = LearningCurve(self.fpath)
        lc.parse()
        loss = lc.list('loss')
        acc = lc.list('accuracy')
        assert_equal(loss.shape, acc.shape)
        assert_false(np.all(loss == acc))

    def test_list_invalid_key(self):

        lc = LearningCurve(self.fpath)
        lc.parse()
        assert_raises(KeyError, lc.list, 'wrong-key', phase=Phase.TRAIN)
        assert_raises(KeyError, lc.list, 'wrong-key', phase=Phase.TEST)
        assert_raises(KeyError, lc.list, 'accuracy', phase=Phase.TRAIN)

    def test_name(self):

        lc = LearningCurve(self.fpath)
        assert_is_instance(lc.name(), str)
        assert_greater(len(lc.name()), 0, 'name is empty')

    def test_learning_curve_from_fpath(self):

        lc = LearningCurveFromPath(self.fpath)
        assert_is_not_none(lc)
        train_keys, test_keys = lc.parse()
        assert_list_equal(train_keys, ['NumIters', 'Seconds', 'LearningRate', 'loss'])
        assert_list_equal(test_keys, ['NumIters', 'Seconds', 'LearningRate', 'accuracy', 'loss'])

    def test_learning_curve_from_dir(self):

        lc = LearningCurveFromPath(os.path.split(self.fpath)[0])
        assert_is_not_none(lc)
        train_keys, test_keys = lc.parse()
        assert_list_equal(train_keys, ['NumIters', 'Seconds', 'LearningRate', 'loss'])
        assert_list_equal(test_keys, ['NumIters', 'Seconds', 'LearningRate', 'accuracy', 'loss'])

    def test_learning_curve_from_no_exist(self):

        assert_raises(IOError, LearningCurveFromPath, os.path.join('foo', 'bar'))

    def test_learning_curve_from_dir_empty(self):

        assert_is_none(LearningCurveFromPath(ROOT_PKG_PATH))
