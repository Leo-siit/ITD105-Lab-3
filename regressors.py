from enum import Enum
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
    pass

def elastic_net_view():
    pass

def adaboost_view():
    pass

def knn_view():
    pass

def lasso_view():
    pass

def ridge_view():
    pass

def linear_regression_view():
    pass

def mlp_regressor_view():
    pass

def random_forest_regressor_view():
    pass

def svr_view():
    pass

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
