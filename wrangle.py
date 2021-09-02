import pandas as pd
import os
from env import host, user, password

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Function to establish connection with Codeups MySQL server, drawing username, password, and host from env.py file
def get_db_url(host = host, user = user, password = password, db = 'zillow'):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

# Function to acquire neccessary zillow data from Codeup's MySQL server
def get_zillow_data():
    filename = "zillow.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        SQL = '''
        SELECT bedroomcnt, 
            bathroomcnt, 
            calculatedfinishedsquarefeet, 
            taxvaluedollarcnt, 
            yearbuilt, 
            taxamount, 
            fips
        FROM properties_2017
        WHERE propertylandusetypeid = 261
        '''
        # read the SQL query into a dataframe
        df = pd.read_sql(SQL, get_db_url())
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)
        # Return the dataframe to the calling code
        return df

# Useful functions to provide high-level overview of the data
def object_vals(df):
    '''
    This is a helper function for viewing the value_counts for object cols.
    '''
    for col, vals in df.iteritems():
        print(df[col].value_counts(dropna = False))

def col_range(df):
    stats_df = df.describe().T
    stats_df['range'] = stats_df['max'] - stats_df['min']
    return stats_df

def summarize_df(df):
    '''
    This function returns the shape, info, a preview, the value_counts of object columns
    and the summary stats for numeric columns.
    '''
    print(f'This dataframe has {df.shape[0]} rows and {df.shape[1]} columns.')
    print('------------------------')
    print('')
    print(df.info())
    print('------------------------')
    print('')
    object_vals(df)
    print('------------------------')
    print('')
    print(col_range(df))