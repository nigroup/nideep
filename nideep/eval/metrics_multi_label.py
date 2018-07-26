'''
Created on Apr 6, 2016

@author: marcenacp
'''
import h5py
import numpy as np
from nideep.eval.inference import infer_to_h5_fixed_dims
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, hamming_loss

def get_inputs_outputs(fpath_h5, key_label, key_score, threshold=0):
    """
    Retrieve true labels and score from hdf5 inference file
    Derive predictions according to the given threshold
    Output true labels, score and predictions
    """
    with h5py.File(fpath_h5, "r") as f:
        y_true = np.squeeze(f[key_label][:])
        if y_true.ndim == 1:
            dim = y_true.shape[0]
            y_true = y_true.reshape((1, dim))
        y_score = np.squeeze(f[key_score][:])
        y_pred = np.array([[prob>=threshold for prob in preds] for preds in y_score])
    return y_true, y_score, y_pred

def example_based_measures(fpath_h5, key_label, key_score, threshold):
    """
    Evaluation measures used to assess the predictive performance in example-based
    learning: macro/micro precision, macro/micro recall and macro/micro f1
    """
    m = {}
    y_true, _, y_pred = get_inputs_outputs(fpath_h5, key_label, key_score, threshold)
    for cat in ['macro', 'micro']:
        m[cat+'_precision'], m[cat+'_recall'], m[cat+'_f1'], _ = precision_recall_fscore_support(y_true, y_pred, average=cat)
    return m

def label_based_measures(fpath_h5, key_label, key_score, threshold=0):
    """
    Evaluation measures used to assess the predictive performance in multi-label
    label-based learning: hamming_loss, precision, recall and f1
    """
    m = {}
    y_true, _, y_pred = get_inputs_outputs(fpath_h5, key_label, key_score, threshold)
    m['hamming_loss'] = hamming_loss(y_true, y_pred)
    m['precision'], m['recall'], m['f1'], _ = precision_recall_fscore_support(y_true, y_pred)
    return m

def all_measures(fpath_h5, key_label, key_score, threshold=0):
    """
    All measures, both label-based and example-based
    """
    m1 = example_based_measures(fpath_h5, key_label, key_score, threshold)
    m2 = label_based_measures(fpath_h5, key_label, key_score, threshold)
    return dict(m1.items() + m2.items())

def evaluation_measure_per_class(evaluation_measure, fpath_h5, key_label, key_score, threshold=0):
    """
    Evaluation measure for each class
    """
    m = {}
    y_true, _, y_pred = get_inputs_outputs(fpath_h5, key_label, key_score, threshold)
    classes = range(y_true.shape[1])
    for c in classes:
        m[c] = evaluation_measure(y_true[:,c], y_pred[:,c])
    return m

def confusion_matrix_per_class(fpath_h5, key_label, key_score, threshold=0):
    """
    Confusion matrix for each class
    """
    return evaluation_measure_per_class(confusion_matrix, fpath_h5, key_label, key_score, threshold)

def accuracy_per_class(fpath_h5, key_label, key_score, threshold=0):
    """
    Accuracy for each class
    """
    return evaluation_measure_per_class(accuracy_score, fpath_h5, key_label, key_score, threshold)
