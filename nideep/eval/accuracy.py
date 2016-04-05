import numpy as np
from sklearn.metrics import hamming_loss

def hamming_accuracy_from_net(net, key_label, key_score, threshold):
    """
    Run hamming accuracy  on a given net
    """
    solver.test_nets[0].forward()
    y_true = solver.test_nets[0].blobs['label'].data
    y_prob = solver.test_nets[0].blobs['score'].data
    return hamming_accuracy_from_blob(y_true, y_prob, threshold)

def hamming_accuracy_from_blob(y_true, y_prob, threshold):
    """
    Run hamming accuracy on a given blob
    """
    y_prob = np.squeeze(y_prob)
    y_pred = np.array([[prob>=threshold=0 for prob in preds] for preds in y_prob])
    return hamming_accuracy(y_true, y_pred)

def hamming_accuracy(y_true, y_pred):
    """
    hamming_accuracy = 1 - hamming_loss
    """
    return 1 - hamming_loss(y_true, y_pred)
