from sklearn.model_selection import train_test_split
import numpy as np

class TrainTestSplit:
    def __init__(self, test_size=0.25, random_state=None):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        train_idx, test_idx = train_test_split(
            np.arange(len(X)), 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1
