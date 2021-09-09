import pandas as pd
import wrangle as wr


import itertools
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

###################################         HELPER FUNCS         ###################################

def separate_feats(df):
    ''' 
    Creates a combination of all possible features as well as separates quant vars and cat vars
    '''

    # Filters for non-object and categorical vars
    cat_feats = [col for col in df.columns if (df[f'{col}'].dtype != object) & (df[f'{col}'].nunique() <= 8)] 

    # Filters for quantitative variables
    quant_feats = [col for col in df.columns if (df[f'{col}'].dtype != object) & (df[f'{col}'].nunique() > 8)]

    # General features that are not objects
    feats = [col for col in df.columns if df[f'{col}'].dtype != object]

    return feats, cat_feats, quant_feats


def pairing(df):
    ''' 
    Helper function for vizualizations, creates quant/cat var pairs. 
    Takes in a dataframe and outputs a list of unique pairs and quant/cat var pairings
    '''
    # Separating features
    feats, cat_feats, quant_feats = separate_feats(df)

    # Creating raw pairs of all features
    pairs = []
    pairs.extend(list(itertools.product(cat_feats, quant_feats)))
    pairs.extend(list(itertools.product(cat_feats, feats)))
    pairs.extend(list(itertools.product(quant_feats, feats)))
    
    # Whittling down pairs to unique combos of quant and cat vars
    unique_pairs = []
    cat_quant_pairs = []
    for pair in pairs:
        if pair[0] != pair[1]:
            if pair not in unique_pairs:
                if (pair[1], pair[0]) not in unique_pairs:
                    unique_pairs.append(pair)
                    if (pair[0] not in quant_feats) & (pair[1] not in cat_feats):
                        cat_quant_pairs.append(pair)


    return pairs, unique_pairs, cat_quant_pairs


def feature_combos(df):
    ''' 
    Creates a list of all possible feature combinations
    '''
    feats, cat_feats, quant_feats = separate_feats(df)
    combos = []
    for i in range(2, len(feats) + 1):
        combos.extend(list(itertools.combinations(feats, i)))
    return combos
    

def hist_combos(df):
    ''' Create a histplot for vars
    '''
    feats, cat_feats, quant_feats = separate_feats(df)
    
    plt.figure(figsize=(16, 3))

    for i, feat in enumerate(feats):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(feats), plot_number)

        # Title with column name.
        plt.title(feat)

        # Display histogram for column.
        df[feat].hist(bins=5)

        # Hide gridlines.
        plt.grid(False)
    
        # turn off scientific notation
        plt.ticklabel_format(useOffset=False)
    plt.tight_layout()
    plt.show()


def relplot_pairs(df):
    ''' 
    Plots each unique cat/quant pair using a relplot
    '''
    pairs, u_pairs, cq_pairs = pairing(df)
    for i, pair in enumerate(cq_pairs):
        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 
        # subplotting based on index
        plt.subplot(1, len(cq_pairs), plot_number)
        sns.relplot(x= pair[0], y= pair[1], data=df)
        plt.title(f'{pair[0]} vs {pair[1]}')
        plt.show()
        print('_____________________________________________')


def lmplot_pairs(df):
    ''' 
    Plots each unique cat/quant pair using a lmplot
    '''
    pairs, u_pairs, cq_pairs = pairing(df)
    for pair in cq_pairs:
        sns.lmplot(x= pair[0], y= pair[1], data=df, line_kws={'color': 'red'})
        plt.title(f'{pair[0]} vs {pair[1]}')
        plt.show()
        print('_____________________________________________')


def jointplot_pairs(df):
    ''' 
    Plots each unique cat/quant pair using a jointplot
    '''
    pairs, u_pairs, cq_pairs = pairing(df)
    for pair in cq_pairs:
        sns.jointplot(x= pair[0], y= pair[1], data=df, kind = 'reg', height = 4)
        plt.title(f'{pair[0]} vs {pair[1]}')
        plt.tight_layout()
        plt.show()
        print('_____________________________________________')


def pairplot_combos(df):
    ''' 
    Plots combinations of quant variables using pairplots where combo length greater or equal to 2
    '''

    # Get list of feats
    feats, cat_feats, quant_feats = separate_feats(df)
    quant = df[quant_feats]

    # From quantitative vars get combos to create pairplots of
    combos = feature_combos(quant)
    for combo in combos:
        plt.figure(figsize=(5,5))
        sns.pairplot(quant, corner=True)
        plt.title(f'Pairplot of: {combo}')
        plt.tight_layout()
        plt.show()
        print('_____________________________________________')


def heatmap_combos(df):
    ''' 
    Create a heatmaps for unique combos of vars where combo length is greater than 3
    '''
    feats, cat_feats, quant_feats = separate_feats(df)
    combos = feature_combos(df)
    for combo in combos:
        if len(combo) > 3:
            plt.figure(figsize=(5,5))
            plt.title(f'Heatmap of {len(combo)} features')
            sns.heatmap(df[list(combo)].corr(), cmap = 'plasma', annot=True)
            plt.tight_layout()
            plt.show()
            print(f'Heatmap features: {combo}')
            print('_____________________________________________')


###################################         EXPOLORE FUNCS         ###################################

def plot_categorical_and_continuous_vars(df):
    print('Histograms for each feature')
    hist_combos(df)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Relplots for each cat/quant feature pairing')
    relplot_pairs(df)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Lmplots for each cat/quant feature pairing')
    lmplot_pairs(df)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Jointplots for each cat/quant feature pairing')
    jointplot_pairs(df)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Heatmaps for feature combos (len > 3)')
    heatmap_combos(df)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Pairplots for quantitative feature combos')
    pairplot_combos(df)