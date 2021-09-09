import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

from scipy import stats

from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression 
from math import sqrt

import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import evaluate
import math

###################################         MODEL FUNCS         ###################################

def lin_reg(df, x, y):
    '''
    Creates and fits a linear regression model. 
    Takes in a dataframe, driver of target var (x), and the target var (y)
    Returns the dataframe with columns:
    - ŷ (predictions of the target (Y) based upon driver (X))
    - baseline (baseline predictions of Y)
    '''

    # Create model
    model = LinearRegression(normalize = True)

    # Fitting the model using x and y
    model.fit(df[[x]], df[y])

    # Creating ŷ predictions column
    df['ŷ'] = model.predict(df[[x]])

    # Create baseline predictions column
    df['baseline'] = df[y].mean()
    
    return df


def plot_residuals(y, ŷ):
    residuals = y - ŷ
    plt.hlines(0, y.min(), y.max(), ls=':')
    plt.scatter(y, residuals)
    plt.ylabel('residual ($y - \hat{y}$)')
    plt.xlabel('actual value ($y$)')
    plt.title('Actual vs Residual')
    plt.show()

###################################         MATH EVAL         ###################################

def residuals(y, ŷ):
    return y - ŷ

def sse(y, ŷ):
    return (residuals(y, ŷ) **2).sum()

def mse(y, ŷ):
    n = y.shape[0]
    return sse(y, ŷ) / n

def rmse(y, ŷ):
    return math.sqrt(mse(y, ŷ))

def ess(y, ŷ):
    return ((ŷ - y.mean()) ** 2).sum()

def tss(y):
    return ((y - y.mean()) ** 2).sum()

def r2_score(y, ŷ):
    return ess(y, ŷ) / tss(y)


def regression_errors(y, ŷ):
    return pd.Series({
        'sse': sse(y, ŷ),
        'ess': ess(y, ŷ),
        'tss': tss(y),
        'mse': mse(y, ŷ),
        'rmse': rmse(y, ŷ),
    })

def baseline_mean_errors(y):
    predicted = y.mean()
    return {
        'sse': sse(y, ŷ),
        'mse': mse(y, ŷ),
        'rmse': rmse(y, ŷ),
    }

def better_than_baseline(y, ŷ):
    rmse_baseline = rmse(y, y.mean())
    rmse_model = rmse(y, ŷ)
    if rmse_model < rmse_baseline:
        return 'Better than baseline'
    elif rmse_model == rmse_baseline:
        return 'Same as baseline'
    else:
        return 'Worse than baseline'

###################################         AUTO-FEATURE SELECTION         ###################################

def kbest_features(df, target, k, stratify = False):
    '''
    df : Unscaled dataframe
    target : Target variable of df
    k : Number of features to select
    stratify : Default is false. If true, stratifies during the train/test split
    '''
    
    # only selects numeric cols and separates target
    X = df[[col for col in df.columns if df[col].dtype != object]].drop(columns = target)
    y = df[target]

    # train, test split checking for stratify
    if stratify == True:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, 
                                                            random_state=123,
                                                            stratify=df[target])
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
                                                            random_state = 123)

    # scaling data
    scaler = StandardScaler()

    # fitting scaler to each split
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # creating SelectKBest object with {k} selected features
    kbest = SelectKBest(f_regression, k= k)
    
    # fitting object
    kbest.fit(x_scaled, y)
    
    # assigning features to var
    features = x.columns[kbest.get_support()]
    
    return features


def rfe_features(x_scaled, x, y, n, model = LinearRegression()):
    '''
    X_scaled : Takes in a scaled dataframe of features not including the target
    X : Unscaled dataframe without target feature
    Y : Takes in an an array containing the target
    N : Number of features to whittle down to
    Model: Which model type to use for RFE
    '''
    # creating RFE object with {n} selected features
    rfe = RFE(estimator= model, n_features_to_select=n)
    
    # fitting object
    rfe.fit(x_scaled, y)
    
    # assigning features to var
    features = x.columns[rfe.get_support()]
    
    return features