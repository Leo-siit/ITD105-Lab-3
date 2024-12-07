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

def decision_tree_view(prefix="Decision_Tree"):
    st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key=f"{prefix}_test_size")
    st.slider("Random Seed", 1, 100, 50, key=f"{prefix}_random_seed")
    st.slider("Max Depth", 1, 20, 5, key=f"{prefix}_max_depth")
    st.slider("Min Samples Split", 2, 10, 2, key=f"{prefix}_min_samples_split")
    st.slider("Min Samples Leaf", 1, 10, 1, key=f"{prefix}_min_samples_leaf")

def gaussian_nb_view(prefix="Gaussian_NB"):
    st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key=f"{prefix}_test_size")
    st.slider("Random Seed", 1, 100, 7, key=f"{prefix}_random_seed")
    st.number_input("Var Smoothing (Log Scale)", min_value=-15, max_value=-1, value=-9, step=1, key=f"{prefix}_var_smoothing")

def adaboost_view(prefix="AdaBoost"):
    st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key=f"{prefix}_test_size")
    st.slider("Random Seed", 1, 100, 42, key=f"{prefix}_random_seed")
    st.slider("Regularization Parameter (C)", 0.1, 10.0, 1.0, key=f"{prefix}_C")
    st.selectbox("Kernel Type", options=['linear', 'poly', 'rbf', 'sigmoid'], key=f"{prefix}_kernel")

def knn_view(prefix="KNN"):
    st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key=f"{prefix}_test_size")
    st.slider("Random Seed", 1, 100, 7, key=f"{prefix}_random_seed")
    st.slider("Number of Neighbors", 1, 20, 5, key=f"{prefix}_n_neighbors")
    st.selectbox("Weights", options=["uniform", "distance"], key=f"{prefix}_weights")
    st.selectbox("Algorithm", options=["auto", "ball_tree", "kd_tree", "brute"], key=f"{prefix}_algorithm")

def logistic_regression_view(prefix="Logistic_Regression"):
    st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key=f"{prefix}_test_size")
    st.slider("Random Seed", 1, 100, 7, key=f"{prefix}_random_seed")
    st.slider("Max Iterations", 100, 500, 200, key=f"{prefix}_max_iter")
    st.selectbox("Solver", options=["lbfgs", "liblinear", "sag", "saga", "newton-cg"], key=f"{prefix}_solver")
    st.number_input("Inverse of Regularization Strength", min_value=0.01, max_value=10.0, value=1.0, key=f"{prefix}_C")

def mlp_classifier_view(prefix="MLP_Classifier"):
    st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key=f"{prefix}_test_size")
    st.slider("Random Seed", 1, 100, 7, key=f"{prefix}_random_seed")
    st.text_input("Hidden Layer Sizes (e.g., 65,32)", "65,32", key=f"{prefix}_hidden_layer_sizes")
    st.selectbox("Activation Function", options=["identity", "logistic", "tanh", "relu"], key=f"{prefix}_activation")
    st.slider("Max Iterations", 100, 500, 200, key=f"{prefix}_max_iter")

def perceptron_classifier_view(prefix="Perceptron"):
    st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key=f"{prefix}_test_size")
    st.slider("Random Seed", 1, 100, 7, key=f"{prefix}_random_seed")
    st.slider("Max Iterations", 100, 500, 200, key=f"{prefix}_max_iter")
    st.number_input("Initial Learning Rate", min_value=0.001, max_value=10.0, value=1.0, key=f"{prefix}_eta0")
    st.number_input("Tolerance for Stopping Criterion", min_value=0.0001, max_value=1.0, value=1e-3, key=f"{prefix}_tol")

def random_forest_view(prefix="Random_Forest"):
    st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key=f"{prefix}_test_size")
    st.slider("Random Seed", 1, 100, 7, key=f"{prefix}_random_seed")
    st.slider("Number of Estimators (Trees)", 10, 200, 100, key=f"{prefix}_n_estimators")
    st.slider("Max Depth of Trees", 1, 50, None, key=f"{prefix}_max_depth")
    st.slider("Min Samples to Split a Node", 2, 10, 2, key=f"{prefix}_min_samples_split")
    st.slider("Min Samples in Leaf Node", 1, 10, 1, key=f"{prefix}_min_samples_leaf")

def svm_view(prefix="SVM"):
    st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key=f"{prefix}_test_size")
    st.slider("Random Seed", 1, 100, 42, key=f"{prefix}_random_seed")
    st.slider("Regularization Parameter (C)", 0.1, 10.0, 1.0, key=f"{prefix}_C")
    st.selectbox("Kernel Type", options=['linear', 'poly', 'rbf', 'sigmoid'], key=f"{prefix}_kernel")

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
