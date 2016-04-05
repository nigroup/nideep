import numpy as np
from sklearn.metrics import hamming_loss

def hamming_accuracy_from_net(net, key_label, key_score, threshold=0):
    """
    Run hamming accuracy  on a given net
    """
    net.forward()
    y_true = net.blobs[key_label].data
    y_prob = net.blobs[key_score].data
    return hamming_accuracy_from_blob(y_true, y_prob, threshold)

def hamming_accuracy_from_blob(y_true, y_prob, threshold=0):
    """
    Run hamming accuracy on a given blob
    """
    # Normalize blob matrix to a 2D array
    y_true = np.squeeze(y_true)
    if y_true.ndim == 1:
        dim = y_true.shape[0]
        y_true = y_true.reshape((1, dim))
    # Generate predictions
    y_pred = np.array([[prob>=threshold for prob in preds] for preds in y_prob])
    return hamming_accuracy(y_true, y_pred)

def hamming_accuracy(y_true, y_pred):
    """
    hamming_accuracy = 1 - hamming_loss
    """
    return 1 - hamming_loss(y_true, y_pred)
