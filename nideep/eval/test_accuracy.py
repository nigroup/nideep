from nose.tools import assert_equals
import accuracy
import numpy as np

class TestHammingAccuracy:
    def test_ideal_situation(self):
        """
        Ideal case with accuracy 1
        """
        for n in range(2, 100):
            y_true = np.identity(n)
            y_pred = np.identity(n)
            assert_equals(accuracy.hamming_accuracy(y_true, y_pred), 1)

    def test_all_wrong(self):
        """
        Wrong classification with zero accuracy
        """
        for n in range(2, 100):
            y_true = np.ones((n,n))
            y_pred = np.zeros((n,n))
            assert_equals(accuracy.hamming_accuracy(y_true, y_pred), 0)

    def test_compare_to_sklearn(self):
        """
        Comparison using calculations by hand
        """
        for n in range(2, 100):
            # Multi-label classification notation with binary label indicators
            y_true = np.random.randint(2, size=(n,n))
            y_pred = np.random.randint(2, size=(n,n))
            # Hamming accuracy by hand
            hamming_loss = 0.
            for i in range(n):
                for j in range(n):
                    hamming_loss += (y_true[i,j] != y_pred[i,j])
            # Compare both
            acc_sklearn = 1. - hamming_loss/n**2
            acc_hamming = accuracy.hamming_accuracy(y_true, y_pred)
            assert_equals(acc_hamming, acc_sklearn)

class TestHammingAccuracyFromBlob:
    def test_accuracy_1(self):
        """
        Extreme case with accuracy 1 and no threshold
        """
        for n in range(2, 100):
            y_true = np.ones((n,n))
            y_prob = np.ones((n,n))/n
            assert_equals(accuracy.hamming_accuracy_from_blob(y_true, y_prob), 1)

    def test_accuracy_0(self):
        """
        Extreme case with accuracy 0 and no threshold
        """
        for n in range(2, 100):
            y_true = np.ones((n,n))
            y_prob = - np.ones((n,n))/n
            assert_equals(accuracy.hamming_accuracy_from_blob(y_true, y_prob), 0)

    def test_accuracy_1_with_threshold(self):
        """
        Accuracy 1 if all values above thresold
        """
        threshold = -.1
        for n in range(2, 100):
            y_true = np.ones((n,n))
            y_prob = np.ones((n,n))/n
            assert_equals(accuracy.hamming_accuracy_from_blob(y_true, y_prob, threshold), 1)

    def test_accuracy_1_with_threshold(self):
        """
        Accuracy 0 if all values under threshold
        """
        threshold = 1
        for n in range(2, 100):
            y_true = np.ones((n,n))
            y_prob = np.ones((n,n))/n
            assert_equals(accuracy.hamming_accuracy_from_blob(y_true, y_prob, threshold), 0)

    def test_extra_dimension(self):
        """
        If y_true has extra dimensions, we use a regular compact matricial form in 2D.
        """
        y_true = np.array([[[[1]],
                            [[0]],
                            [[0]]],
                           [[[0]],
                            [[1]],
                            [[0]]],
                           [[[0]],
                            [[0]],
                            [[1]]],])
        y_true_compare = np.identity(3)
        y_prob = np.array([[+1, -1, -1],
                           [-1, +1, -1],
                           [-1, -1, +1]])
        assert_equals(accuracy.hamming_accuracy_from_blob(y_true, y_prob),
                      accuracy.hamming_accuracy_from_blob(y_true_compare, y_prob))

    def test_singleton_batch(self):
        """
        In the case of a batch size of 1, y_true has shape (number_of_classes,) and accuracy should be correctly computed.
        """
        y_true = np.ones((10,))
        y_true_compare = np.ones((1,10))
        y_prob = - np.ones((1, 10))
        assert_equals(accuracy.hamming_accuracy_from_blob(y_true, y_prob),
                      accuracy.hamming_accuracy_from_blob(y_true_compare, y_prob))
