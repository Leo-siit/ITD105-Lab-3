from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import streamlit as st


MODEL_MAPPING = {
    "Decision Tree": DecisionTreeClassifier,
    "Gaussian Naive Bayes": GaussianNB,
    "AdaBoost": AdaBoostClassifier,
    "K-Nearest Neighbors": KNeighborsClassifier,
    "Logistic Regression": LogisticRegression,
    "MLP Classifier": MLPClassifier,
    "Perceptron Classifier": Perceptron,
    "Random Forest": RandomForestClassifier,
    "Support Vector Machine (SVM)": SVC,
}

def create_model(model_type, **kwargs):
    model_class = MODEL_MAPPING.get(model_type)
    if model_class is None:
        raise ValueError('Invalid model type')
    return model_class(**kwargs)

def score_model(model, X, Y, cv):
    return cross_val_score(model, X, Y, cv=cv, scoring='accuracy').mean()

def decision_tree_view():
    with st.container():
        test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
        random_seed = st.slider("Random Seed", 1, 100, 50)
        max_depth = st.slider("Max Depth", 1, 20, 5)
        min_samples_split = st.slider("Min Samples Split", 2, 10, 2)
        min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1)
    # st.session_state.

def gaussian_nb_view():
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test_size")
    random_seed = st.slider("Random Seed", 1, 100, 7, key="random_seed")
    var_smoothing = st.number_input("Var Smoothing (Log Scale)", min_value=-15, max_value=-1, value=-9, step=1, key="var_smoothing")

def adaboost_view():
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test8")
    random_seed = st.slider("Random Seed", 1, 100, 42, key="seed8")
    C = st.slider("Regularization Parameter (C)", 0.1, 10.0, 1.0)
    kernel = st.selectbox("Kernel Type", options=['linear', 'poly', 'rbf', 'sigmoid'])

def knn_view():
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="keytest3")
    random_seed = st.slider("Random Seed", 1, 100, 7, key="seed3")
    n_neighbors = st.slider("Number of Neighbors", 1, 20, 5)
    weights = st.selectbox("Weights", options=["uniform", "distance"])
    algorithm = st.selectbox("Algorithm", options=["auto", "ball_tree", "kd_tree", "brute"])

def logistic_regression_view():
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test4")
    random_seed = st.slider("Random Seed", 1, 100, 7, key="seed4")
    max_iter = st.slider("Max Iterations", 100, 500, 200)
    solver = st.selectbox("Solver", options=["lbfgs", "liblinear", "sag", "saga", "newton-cg"])
    C = st.number_input("Inverse of Regularization Strength", min_value=0.01, max_value=10.0, value=1.0)

def mlp_classifier_view():
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test5")
    random_seed = st.slider("Random Seed", 1, 100, 7, key="seed5")
    hidden_layer_sizes = st.text_input("Hidden Layer Sizes (e.g., 65,32)", "65,32")
    activation = st.selectbox("Activation Function", options=["identity", "logistic", "tanh", "relu"])
    max_iter = st.slider("Max Iterations", 100, 500, 200, key="max5")

def perceptron_classifier_view():
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test6")
    random_seed = st.slider("Random Seed", 1, 100, 7, key="seed6")
    max_iter = st.slider("Max Iterations", 100, 500, 200, key="max6")
    eta0 = st.number_input("Initial Learning Rate", min_value=0.001, max_value=10.0, value=1.0)
    tol = st.number_input("Tolerance for Stopping Criterion", min_value=0.0001, max_value=1.0, value=1e-3)

def random_forest_view():
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test7")
    random_seed = st.slider("Random Seed", 1, 100, 7, key="seed7")
    n_estimators = st.slider("Number of Estimators (Trees)", 10, 200, 100)
    max_depth = st.slider("Max Depth of Trees", 1, 50, None)  # Allows None for no limit
    min_samples_split = st.slider("Min Samples to Split a Node", 2, 10, 2)
    min_samples_leaf = st.slider("Min Samples in Leaf Node", 1, 10, 1)

def svm_view():
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test8")
    random_seed = st.slider("Random Seed", 1, 100, 42, key="seed8")
    C = st.slider("Regularization Parameter (C)", 0.1, 10.0, 1.0)
    kernel = st.selectbox("Kernel Type", options=['linear', 'poly', 'rbf', 'sigmoid'])

MODEL_PARAMS = {
    "Decision Tree": decision_tree_view,
    "Gaussian Naive Bayes": gaussian_nb_view,
    "AdaBoost": adaboost_view,
    "K-Nearest Neighbors": knn_view,
    "Logistic Regression": logistic_regression_view,
    "MLP Classifier": mlp_classifier_view,
    "Perceptron Classifier": perceptron_classifier_view,
    "Random Forest": random_forest_view,
    "Support Vector Machine (SVM)": svm_view,
}
