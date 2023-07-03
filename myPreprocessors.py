import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

#Clase para manejo de variables temporales en el modelo de House Price
class TremporalVariableTransformer(BaseEstimator, TransformerMixin):

    #Constructor
    def __init__(self, variables, reference_variable):
        if not isinstance(variables, list):
            raise ValueError("Las varibles debe ser incluida en una lista.")
        
        self.variables = variables
        self.reference_variable = reference_variable

    #metodo fit para habilitar metodo transform
    def fit(self, X, y=None):
        return self

    #metodo para transformar variables temporales.
    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[self.reference_variable] - X[feature]
        return X

#===========================================================================
#Clase para transformación de variables categóricas ordinales.
class Mapper(BaseEstimator, TransformerMixin):
    
    #constructor
    def __init__(self, variables, mappings):
        
        if not isinstance(variables, list):
            raise ValueError("Las varibles debe ser incluida en una lista.")
        
        #campos de clase Mapper.
        self.variables = variables
        self.mappings = mappings
        
    #Metodo Fit
    def fit(self, X, y=None):
        #fit no hace nada, pero es requisito para el pipeline
        return self
    
    #Metodo transform
    def transform(self, X):
        X = X.copy()
        for variable in self.variables:
            X[variable] = X[variable].map(self.mappings)
        return X
#===========================================================================
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            freq_encoder_dict = X[var].value_counts().to_dict()
            X[var] = X[var].map(freq_encoder_dict)
        return X


#===========================================================================
class OutlierDetector(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            LI, LS = self.detectOutliersLimits(X, var)
            X[var] = np.where(X[var] > LS, LS, np.where(X[var] < LI, LI, X[var]))
        return X
    
    def detectOutliersLimits(self, dataset, col):
        IQR = dataset[col].quantile(0.75) - dataset[col].quantile(0.25)
        LI =  dataset[col].quantile(0.25) - (IQR * 1.75)
        LS = dataset[col].quantile(0.75) + (IQR * 1.75)
        return LI, LS

