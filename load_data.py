#!/usr/bin/python

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

import datetime
from dateutil.parser import parse

def clean_data(indf):
    """
    clean train/test dataframes
    """
    indf['Open Date'] = indf['Open Date'].apply(lambda x: parse(x).date())
    indf['Open Date'] = indf['Open Date']\
                                .apply(lambda x: 
                                        (x-datetime.date(year=1995, month=1,
                                                         day=1)).days)

    indf['City Group'] = indf['City Group']\
                            .map({'Other': 0, 'Big Cities': 1})
    ### Type == 'MB' doesn't exist in training set...    
    types = {'FC': 0, 'IL': 1, 'DT': 2, 'MB': 3}
    indf['Type'] = indf['Type'].map(types)
    for idx in range(4):
        indf['Type%d' % idx] = (indf['Type'] == idx).astype(np.int64)

    ### can't predict something which doesn't exist in training set...
    indf = indf.drop(labels=['City', 'Type', 'Type1', 'Type2', 
                             'Type3'], axis=1)

    return indf

def load_data(do_drop_list=False, do_plots=False):
    train_df = pd.read_csv('train.csv.gz', compression='gzip')
    test_df = pd.read_csv('test.csv.gz', compression='gzip')
    submit_df = pd.read_csv('sampleSubmission.csv.gz', compression='gzip')

    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    print train_df.columns
    print test_df['City Group'].describe()

#    print train_df['revenue'].describe()
#    for col in test_df.columns:
#        print '\'%s\': [%d, %d, 0],' % (col, min(train_df[col].min(), test_df[col].min()), \
#                   max(train_df[col].max(), test_df[col].max()))

    if do_plots:
        from plot_data import plot_data
        plot_data(train_df, prefix='html_train')
        plot_data(test_df, prefix='html_test')

    ### wanted to keep track of feature_list
    feature_list = train_df.drop(['Id', 'revenue'], axis=1).columns
    print 'features', list(feature_list)


    xtrain = train_df.drop(labels=['Id', 'revenue'], axis=1).values
    ytrain = train_df['revenue'].values
    xtest = test_df.drop(labels=['Id'], axis=1).values
    ytest = submit_df
    return xtrain, ytrain, xtest, ytest, feature_list

if __name__ == '__main__':
    xtrain, ytrain, xtest, ytest, feature_list = load_data(do_plots=False)
    
    print [v.shape for v in xtrain, ytrain, xtest, ytest]
