from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb


class Classifier(BaseEstimator):
    def __init__(self):
        self.model = xgb.XGBClassifier()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        y_pred = self.model.predict_proba(X)
        return y_pred


def get_estimator():
    num_cols =  ['age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)', 'eyesight(right)', 
                          'hearing(left)', 'hearing(right)', 'systolic', 'relaxation', 'fasting blood sugar', 'Cholesterol', 
                          'triglyceride', 'HDL', 'LDL', 'hemoglobin', 'Urine protein', 'serum creatinine', 'AST', 'ALT', 
                          'Gtp']
    cat_cols =  ['gender', 'tartar','dental caries' ]
  
    #Preprocessing for numerical columns in the dataset

    #Let us note that we do not need to do the imputer since our dataset does not have missing values.


    num_pipeline = Pipeline(steps=[
      ('scaler', StandardScaler())
    ])  


    #Preprocessing categorical columns in the dataset 
    cat_pipeline = Pipeline(steps=[
      ('encoder', OneHotEncoder(handle_unknown='ignore')) 
    ])

    # We define the column transformer to apply the appropriate preprocessing to each column
    preprocessor = ColumnTransformer(transformers=[
      ('num', num_pipeline, num_cols),
      ('cat', cat_pipeline, cat_cols)
    ])

    classifier = Classifier()

    pipe = make_pipeline( preprocessor, classifier)
    return pipe