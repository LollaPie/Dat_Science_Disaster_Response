import pandas as pd
import MLSMOTE
from sklearn.base import BaseEstimator, TransformerMixin


class Sampler(BaseEstimator):
    """
    Resampler to pass augmented data in a imblearn pipeline
    """
    
    def __init__(self, augmented_sets):
        self.augmented_sets = augmented_sets
        
    def fit_resample(self, X, y):
        return self.resample(X, y)
        
    def resample(self, X_tfidf, y):
        # Getting minority samples of the dataframe
        X_sub, y_sub = MLSMOTE.get_minority_samples(X_tfidf, y)

        # Generating synthetic samples based on the minority samples
        X_res, y_res = MLSMOTE.MLSMOTE(X_sub, y_sub, 500)
        
        X_con = pd.concat([X_tfidf, X_res], ignore_index=True)
        y_con = pd.concat([y, y_res], ignore_index=True)
        
        return X_con, y_con


def smote(X, y):
    # Getting minority samples of the dataframe
    X_sub, y_sub = MLSMOTE.get_minority_samples(X, y)

    # Generating synthetic samples based on the minority samples
    X_res, y_res = MLSMOTE.MLSMOTE(X_sub, y_sub, 500)

    X_con = pd.concat([X, X_res], ignore_index=True)
    y_con = pd.concat([y, y_res], ignore_index=True)

    return X_con, y_con