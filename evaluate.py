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
    Takes in a dataframe, driver var (x), and the target var (y)
    Returns the dataframe with columns:
    - y actual
    - ŷ (predictions of the target (Y) based upon driver (X))
    - baseline (baseline predictions of Y)
    - residuals between y actual and ŷ
    - residuals squared
    - residual baseline
    - residual baseline squared
    '''

    # Create model
    model = LinearRegression(normalize = True)

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

def get_metrics(dfo, x, y, raw_data = False):
    ''' will doc later
    '''
    if raw_data == True:
        df1 = lin_reg(dfo, x, y)
        df = df1
    else:
        df = dfo

    # helper var assignmert
    y = df[y]
    x = df[x]
    yhat = df['yhat']
    yhat_baseline = df['yhat_baseline']
    lendf = len(df)
    
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
    df_baseline_eval['baseline_error'] = np.array([SSE_baseline, MSE_baseline, RMSE_baseline])

    # error delta
    df_eval['error_delta'] = df_eval.model_error - df_baseline_eval.baseline_error

    # model significance df
    df_sig = pd.DataFrame(np.array(['ESS', 'TSS', 'R^2']), columns = ['metric'])
    df_sig['model_significance'] = np.array([ESS, TSS, R2])
    # model baseline significance df
    df_baseline_sig = pd.DataFrame(np.array(['ESS_baseline', 'TSS_baseline', 'R^2_baseline']), columns = ['metric'])
    df_baseline_sig['baseline_significance'] = np.array([ESS_baseline, TSS_baseline, R2_baseline])

    df_eval = pd.concat([df_eval, df_baseline_eval], axis = 0)
    df_sig = pd.concat([df_sig, df_baseline_sig], axis = 0)
    return df_eval, df_sig


def plot_residuals(y, ŷ):
    residuals = y - ŷ
    plt.hlines(0, y.min(), y.max(), ls=':')
    plt.scatter(y, residuals)
    plt.ylabel('residual ($y - \hat{y}$)')
    plt.xlabel('actual value ($y$)')
    plt.title('Actual vs Residual')
    plt.show()

###################################         MATH EVAL         ###################################

def residuals(y, ŷ = 'yhat', df = None):
    if df == None:
        return ŷ - y
    else:
        return df[ŷ] - df[y]

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

def t_stat(corr):
    t = (corr * sqrt(n - 2)) / sqrt(1 - corr**2)
    return t

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
