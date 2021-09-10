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
    df['yhat'] = model.predict(df[[x]])

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

def kbest_features(df, target, k, stratify = False, show_scores = False, scaler_type = StandardScaler()):
    '''
    Takes a dataframe and uses SelectKBest to select for
    the most relevant drivers of target.
    
    Parameters:
    -----------
    df : Unscaled dataframe

    target : Target variable of df

    k : Number of features to select

    stratify : No stratification by default. If stratify = true, 
            stratifies for the target during the train/test split

    show_scores : If true, outputs a dataframe containing the top (k) features
            and their respective f-scores.

    scaler_type : Default is StandardScaler, determines the type 
            of scaling applied to the df before
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    Output:
    -------------
    features : A list of features of (k) length that SelectKBest has selected to be the
    main drivers of the target.

    fs_sorted : A sorted dataframe that combines each feature with its respective score.
    '''

    # only selects numeric cols and separates target
    X = df[[col for col in df.columns if df[col].dtype != object]].drop(columns = target)
    y = df[target]
    
    # train, test split checking for stratify
    if stratify == True:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, 
                                                            random_state=123,
                                                            stratify=df[target])
    elif stratify == False:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
                                                            random_state = 123)

    # scaling data
    if scaler_type == StandardScaler():
        scaler = StandardScaler()
    else:
        scaler = scaler_type

    # fitting scaler to each split
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # creating SelectKBest object with {k} selected features
    kbest = SelectKBest(f_regression, k= k)
    
    # fitting object
    kbest.fit(X_train_scaled, y_train)
    
    # assigning features to var
    features = X.columns[kbest.get_support()]
    if show_scores == True:
        # getting feature scores
        scores = kbest.scores_[kbest.get_support()]

        # creating zipped list of feats and their scores
        feat_scores = list(zip(features, scores))
    
        fs_df = pd.DataFrame(data = feat_scores, columns= ['Feat_names','F_Scores'])
    
        fs_sorted = fs_df.sort_values(['F_Scores','Feat_names'], ascending = [False, True])

        return fs_sorted
    else:
        return list(features)


def rfe_features(df, target, n, stratify = False, est_model = LinearRegression(), scaler_type = StandardScaler()):
    '''
    Takes a dataframe and uses Recursive Feature Elimination to select for
    the most relevant drivers of target.
    
    Parameters:
    -----------
    df : Unscaled dataframe
    
    target : Target variable of df
    
    n : Number of features to select
    
    stratify : No stratification by default. If stratify = true, 
            stratifies for the target during the train/test split
            
    est_model : Defailt is LinearRegression, determines the estimator
            used by the RFE function
    
    scaler_type : Default is StandardScaler, determines the type 
            of scaling applied to the df before
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    Output:
    -------------
    A list of features of (n) length that RFE has selected to be the
    main drivers of the target.
    '''
    # only selects numeric cols and separates target
    X = df[[col for col in df.columns if df[col].dtype != object]].drop(columns = target)
    y = df[target]

    # train, test split checking for stratify
    if stratify == True:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, 
                                                            random_state=123,
                                                            stratify=df[target])
    elif stratify == False:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
                                                            random_state = 123)
    
    # scaling data
    if scaler_type == StandardScaler():
        scaler = StandardScaler()
    else:
        scaler = scaler_type
    
    # fitting scaler to each split
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # creating RFE object with {n} selected features
    rfe = RFE(estimator= est_model, n_features_to_select=n)
    
    # fitting object
    rfe.fit(X_train_scaled, y_train)
    
    # assigning features to var
    features = X.columns[rfe.get_support()]
    
    return list(features)