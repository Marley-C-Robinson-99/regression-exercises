import pandas as pd
import os
from env import host, user, password

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

###################################         ACQUIRE         ###################################
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
        LEFT JOIN propertylandusetype 
            USING(propertylandusetypeid)
        WHERE propertylandusedesc 
            IN ("Single Family Residential",                       
                              "Inferred Single Family Residential")
        '''
        # read the SQL query into a dataframe
        df = pd.read_sql(SQL, get_db_url())

        # renaming cols
        df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                              'bathroomcnt':'bathrooms', 
                              'calculatedfinishedsquarefeet':'area',
                              'taxvaluedollarcnt':'tax_value', 
                              'yearbuilt':'year_built',})


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


###################################         PREPARE         ###################################
def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

def prepare_zillow(df):
    ''' Prepare zillow data for exploration'''
    
    # list of non-object cols
    cols = []
    for col, vals in df.iteritems():
        if df[f'{col}'].dtype != object:
            cols.append(col)

    # removing outliers
    df = remove_outliers(df, 1.5, cols)
    
    
    # converting column datatypes
    df.fips = df.fips.astype(object)
    df.year_built = df.year_built.astype(object)
    
    # drop taxamount
    df = df.drop(columns = 'taxamount')
    
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    # impute year built using mode
    imp = SimpleImputer(strategy='most_frequent')  # build imputer

    imp.fit(train[['year_built']]) # fit to train

    # transform the data
    train[['year_built']] = imp.transform(train[['year_built']])
    validate[['year_built']] = imp.transform(validate[['year_built']])
    test[['year_built']] = imp.transform(test[['year_built']])   
    
    return train, validate, test 

def wrangle_zillow():
    '''Acquire and prepare data from Zillow database for explore'''
    train, validate, test = prepare_zillow(get_zillow_data())
    
    return train, validate, test


### Viz ###
def get_hist(df):
    ''' Gets histographs of acquired continuous variables'''
    
    plt.figure(figsize=(16, 3))

    # List of columns
    cols = []
    for col, vals in df.iteritems():
        if df[f'{col}'].dtype != object:
            cols.append(col)

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        df[col].hist(bins=5)

        # Hide gridlines.
        plt.grid(False)

        # turn off scientific notation
        plt.ticklabel_format(useOffset=False)

        plt.tight_layout()

    plt.show()
        
        
def get_box(df):
    ''' Gets boxplots of acquired continuous variables'''
    
    # List of columns
    cols = []
    for col, vals in df.iteritems():
        if df[f'{col}'].dtype != object:
            cols.append(col)

    

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):

        # i starts at 0, but plot should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.boxplot(data=df[[col]])

        # Hide gridlines.
        plt.grid(False)

        # sets proper spacing between plots
        plt.tight_layout()

    plt.show()