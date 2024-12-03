from sklearn.model_selection import KFold, LeaveOneOut, ShuffleSplit
from TrainTestSplit import TrainTestSplit
import streamlit as st

def create_cross_val(cv_type, **kwargs):
    if cv_type == "KFold":
        return KFold(**kwargs)
    elif cv_type == "Leave One Out":
        return LeaveOneOut()
    elif cv_type == "Shuffle Split":
        return ShuffleSplit(**kwargs)
    elif cv_type == "Train Test Split":
        return TrainTestSplit(**kwargs)
    else:
        raise ValueError('Invalid cross validation type')

def kfold_view(container: st, cv_args: dict={}) -> dict:
    n_splits = cv_args.get('n_splits', 5)
    shuffle = cv_args.get('shuffle', False)
    random_state = cv_args.get('random_state', None)

    n_splits = container.number_input('Number of splits', min_value=2, value=n_splits)
    shuffle = container.checkbox('Shuffle', value=shuffle)
    random_state = container.number_input('Random state', min_value=0, value=random_state)
    return {'n_splits': n_splits, 'shuffle': shuffle, 'random_state': random_state}

def leave_one_out_view(container: st, cv_args: dict={}) -> dict:
    container.write('Leave one out does not require any parameters')
    return {}

def shuffle_split_view(container: st, cv_args: dict={}) -> dict:
    n_splits = cv_args.get('n_splits', 5)
    test_size = cv_args.get('test_size', 0.1)
    random_state = cv_args.get('random_state', 42)

    n_splits = container.number_input('Number of splits', min_value=2, value=n_splits)
    # Use only a slider for test size
    test_size = container.slider('Test size', min_value=0.0, max_value=1.0, value=test_size)
    random_state = container.number_input('Random state', value=random_state)
    return {'n_splits': n_splits, 'test_size': test_size, 'random_state': random_state}

def train_test_split_view(container: st, cv_args: dict={}) -> dict:
    test_size = cv_args.get('test_size', 0.1)
    random_state = cv_args.get('random_state', 0)

    # Use only a slider for test size
    test_size = container.slider('Test size', min_value=0.0, max_value=1.0, value=test_size)
    random_state = container.number_input('Random state', min_value=0, value=random_state)
    return {'test_size': test_size, 'random_state': random_state}
