import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

from scipy import stats

from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.feature_selection import f_regression 
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures


import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from evaluate import lin_reg, residuals, sse, mse, rmse, ess, tss, r2_score, t_stat, regression_errors, baseline_mean_errors, better_than_baseline
from wrangle import wrangle_zillow, acquire_zillow, prepare_zillow, scaling

import math

def lin_reg(df, x, y, model_type = LinearRegression(normalize = True)):
    '''
    Creates and fits a linear regression model to a provided dataframe given 
    a driver and target variable.
    
    Parameters
    ----------------
    - df : Dataframe

    - x : driver variable

    - y : target variable

    - model : Default is LinearRegression (OLM) with normalize = True, 
            can use 3 of sklearns linear_models, LinearRegression(OLS), LassoLars, TweedieRegressor (GLM)
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    Outputs
    -------------------
    Returns a dataframe with columns:
    - x variable
    - y actual
    - ŷ (predictions of the target (Y) based upon driver (X))
    - mean_baseline (baseline predictions of Y)
    - residuals between y actual and ŷ
    - residuals squared
    - residual baseline
    - residual baseline squared
    '''

    # Create model
    model = model_type

    # Fitting the model using x and y
    model.fit(df[[x]], df[y])

    df1 = pd.DataFrame(data = df[[x, y]], columns = [f'{x}', f'{y}'])

    # Creating ŷ predictions column
    df1['yhat'] = model.predict(df[[x]])
    yhat = df1['yhat']
    # Create baseline predictions column
    df1['yhat_baseline'] = df[y].mean()

    df1['residuals'] = residuals(df[y], yhat)

    df1['residual^2'] = residuals(df[y], yhat) ** 2

    df1['residual_baseline'] = residuals(df[y], df1['yhat_baseline'])

    df1['residual_baseline^2'] = residuals(df[y], df1['yhat_baseline'])
    
    return df1

def get_metrics(dfo, x, y, raw_data = False, *model_type):
    ''' will doc later
    '''
    if raw_data == True:
        if model_type:
            df1 = lin_reg(dfo, x, y, model_type)
            df = df1
        else:
            df1 = lin_reg(dfo, x, y)
            df = df1
    else:
        df = dfo

    # helper var assignmert
    y = df[y]
    x = df[x]
    yhat = df['yhat']
    yhat_baseline = df['yhat_baseline']
    
    # eval math var assignment
    SSE = sse(y,yhat)
    MSE = mse(y,yhat)
    RMSE = rmse(y,yhat)
    # eval baseline mvars
    SSE_baseline = sse(y, yhat_baseline)
    MSE_baseline = mse(y, yhat_baseline)
    RMSE_baseline = rmse(y, yhat_baseline)

    # model significance vars
    ESS = ess(y,yhat)
    TSS = tss(y)
    R2 = r2_score(y,yhat)

    ESS_baseline = ess(y, yhat_baseline)
    TSS_baseline = ESS_baseline + SSE_baseline
    R2_baseline = r2_score(y, yhat_baseline)
    # eval df
    df_eval = pd.DataFrame(np.array(['SSE','MSE','RMSE']), columns=['metric'])
    df_eval['model_error'] = np.array([SSE, MSE, RMSE])
    # baseline eval df
    df_baseline_eval = pd.DataFrame(np.array(['SSE_baseline','MSE_baseline','RMSE_baseline']), columns=['metric'])
    df_baseline_eval['model_error'] = np.array([SSE_baseline, MSE_baseline, RMSE_baseline])

    # error delta
    df_eval['error_delta'] = df_eval.model_error - df_baseline_eval.model_error

    # model significance df
    df_sig = pd.DataFrame(np.array(['ESS', 'TSS', 'R^2']), columns = ['metric'])
    df_sig['model_significance'] = np.array([ESS, TSS, R2])
    # model baseline significance df
    df_baseline_sig = pd.DataFrame(np.array(['ESS_baseline', 'TSS_baseline', 'R^2_baseline']), columns = ['metric'])
    df_baseline_sig['model_significance'] = np.array([ESS_baseline, TSS_baseline, R2_baseline])

    df_eval = pd.concat([df_eval, df_baseline_eval], axis = 0)
    df_sig = pd.concat([df_sig, df_baseline_sig], axis = 0)
    
    return df_eval, df_sig

###################################         AUTO-FEATURE SELECTION         ###################################

def kbest_features(target, k, show_scores = False, scaler_type = StandardScaler()):
    '''
    Takes a dataframe and uses SelectKBest to select for
    the most relevant drivers of target.
    
    Parameters:
    -----------
    target : Target variable of df

    k : Number of features to select

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
    X_train, X_validate, X_test, y_train, y_validate, y_test = wrangle_zillow(target = target)

    X_train_scaled, X_validate_scaled, X_test_scaled = scaling(X_train, X_validate, X_test)

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