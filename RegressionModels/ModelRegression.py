from sklearn.ensemble import AdaBoostRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor 
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import numpy as np

class ModelClassification:

    self.models = {
        "adaboost": AdaBoostRegressor(),
        "catboost": CatBoostRegressor(allow_writing_files=False),
        "knn": KNeighborsRegressor(),
        "lgbm": LGBMRegressor(),
        "logistic": LogisticRegression(),
        "randomforest": RandomForestRegressor(),
        "xgboost": XGBRegressor(),
        "svr": SVR(),
        "linear": LinearRegression(),
        "linear_ridge": Ridge(),
        "linear_lasso": Lasso(),
        "linear_elastic": ElasticNet(),
        "desicion_tree": DecisionTreeRegressor
    }    

    def __init__(self, X, y, cv):
        self.__X = X
        self.__y = y
        self.__cv = cv

    def __init__(self, X_train, X_test, y_train, y_test):
        self.__X_train = X_train
        self.__X_test = X_test
        self.__y_train = y_train
        self.__y_test = y_test   

    def model_creation(self, model_name, grid_values):
        if grid_values != None:
            self.__clf = GridSearchCV(self.models[model_name], param_grid = grid_values, scoring = 'accuracy')
        else:
            self.__clf = self.models[model_name]

    def predict(self):
        self.__clf.fit(self.__X_train, self.__y_train)
        self.__y_pred = self.__clf.predict(self.__X_test)

    def cross_validation(self):
        self.__scores = cross_val_score(self.__clf, self.__X, self.__y, self.__cv, scoring='accuracy')
    
    def calc_accuracy(self):
        return {
            "MAE" : metrics.mean_absolute_error(self,__y_test, self.__y_pred),
            "MSE" : metrics.mean_squared_error(self.__y_test, self.__y_pred),
            "RMSE" : np.sqrt(metrics.mean_squared_error(self.__y_test, self.__y_pred)),
            "ScoreMean" : self.__scores.mean(),
            "ScoreStd" : self.__scores.std()
        }
    
    def get_y_pred(self):
        self.__y_pred
        
