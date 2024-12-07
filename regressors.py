from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
import streamlit as st

MODEL_MAPPING = {
    "Decision Tree Regressor": DecisionTreeRegressor,
    "Elastic Net": ElasticNet,
    "AdaBoost Regressor": AdaBoostRegressor,
    "K-Nearest Neighbors": KNeighborsRegressor,
    "Lasso": Lasso,
    "Ridge": Ridge,
    "Linear Regression": LinearRegression,
    "MLP Regressor": MLPRegressor,
    "Random Forest Regressor": RandomForestRegressor,
    "Support Vector Regressor (SVR)": SVR,
}

def create_model(model_type, **kwargs):
    model_class = MODEL_MAPPING.get(model_type)
    if model_class is None:
        raise ValueError('Invalid model type')
    return model_class(**kwargs)

def score_model(model, X, Y, cv):
    results = cross_val_score(model, X, Y, cv=cv, scoring='neg_mean_absolute_error')
    mae = -results.mean()
    mae_std = results.std()
    return mae, mae_std

def decision_tree_view():
    st.slider("Max Depth", 1, 20, None, key="dt_max_depth")
    st.slider("Min Samples Split", 2, 20, 2, key="dt_min_samples_split")
    st.slider("Min Samples Leaf", 1, 20, 1, key="dt_min_samples_leaf")

def elastic_net_view():
    st.slider("Alpha (Regularization Strength)", 0.0, 5.0, 1.0, 0.1, key="en_alpha")
    st.slider("L1 Ratio", 0.0, 1.0, 0.5, 0.01, key="en_l1_ratio")
    st.slider("Max Iterations", 100, 2000, 1000, 100, key="en_max_iter")

def adaboost_view():
    st.slider("Number of Estimators", 1, 200, 50, 1, key="ab_n_estimators")
    st.slider("Learning Rate", 0.01, 5.0, 1.0, 0.01, key="ab_learning_rate")

def knn_view():
    st.slider("Number of Neighbors", 1, 20, 5, 1, key="knn_n_neighbors")
    st.selectbox("Weights", ["uniform", "distance"], key="knn_weights")
    st.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree", "brute"], key="knn_algorithm")

def lasso_view():
    st.slider("Regularization Parameter (alpha)", 0.01, 10.0, 1.0, 0.01, key="lasso_alpha")
    st.slider("Maximum Iterations", 100, 1000, 1000, 100, key="lasso_max_iter")

def ridge_view():
    st.slider("Regularization Parameter (alpha)", 0.01, 10.0, 1.0, 0.01, key="ridge_alpha")
    st.slider("Maximum Iterations", 100, 1000, 1000, 100, key="ridge_max_iter")

def linear_regression_view():
    st.text("Linear Regression does not have tunable hyperparameters.")

def mlp_regressor_view():
    st.slider("Hidden Layer Sizes", 10, 200, 100, 10, key="mlp_hidden_layer_sizes")
    st.selectbox("Activation Function", ["identity", "logistic", "tanh", "relu"], key="mlp_activation")
    st.selectbox("Solver", ["adam", "lbfgs", "sgd"], key="mlp_solver")
    st.selectbox("Learning Rate Schedule", ["constant", "invscaling", "adaptive"], key="mlp_learning_rate")
    st.slider("Max Iterations", 100, 2000, 1000, 100, key="mlp_max_iter")
    st.number_input("Random State", value=50, key="mlp_random_state")

def random_forest_regressor_view():
    st.slider("Number of Trees", 10, 500, 100, 10, key="rf_n_estimators")
    st.slider("Max Depth", 1, 50, None, key="rf_max_depth")
    st.slider("Min Samples Split", 2, 10, 2, key="rf_min_samples_split")
    st.slider("Min Samples Leaf", 1, 10, 1, key="rf_min_samples_leaf")
    st.number_input("Random State", value=42, key="rf_random_state")

def svr_view():
    st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"], key="svr_kernel")
    st.slider("Regularization Parameter (C)", 0.01, 100.0, 1.0, 0.01, key="svr_C")
    st.slider("Epsilon", 0.0, 1.0, 0.1, 0.01, key="svr_epsilon")

MODEL_PARAMS = {
    "Decision Tree Regressor": decision_tree_view,
    "Elastic Net": elastic_net_view,
    "AdaBoost Regressor": adaboost_view,
    "K-Nearest Neighbors": knn_view,
    "Lasso": lasso_view,
    "Ridge": ridge_view,
    "Linear Regression": linear_regression_view,
    "MLP Regressor": mlp_regressor_view,
    "Random Forest Regressor": random_forest_regressor_view,
    "Support Vector Regressor (SVR)": svr_view,
}