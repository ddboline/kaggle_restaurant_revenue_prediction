#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Train model, test model, write out submission

@author: Daniel Boline <ddboline@gmail.com>
"""

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFECV

from load_data import load_data

DO_RFECV = False

def score_fn(estimator, x, y):
    """ want to minimze RMSE, maximize 1/RMSE... """
    ypred = estimator.predict(x)
    return 1/np.sqrt(mean_squared_error(ypred, y))

def debug_output(model, feature_list):
    """ some debugging output for the model """
    if hasattr(model, 'grid_scores_'):
        print model.grid_scores_
    if hasattr(model, 'feature_importances_'):
        print 'feature_importances'
        print '\n'.join('%s %s' % \
              (f, i) for f, i in zip(feature_list,
                                     model.feature_importances_))
    if hasattr(model, 'ranking_'):
        print 'drop', [f for f in feature_list
                       if f not in feature_list[model.ranking_ == 1]]
        print 'ranked_features\n'
        print '\n'.join('%s %s' % (f, c) for f, c in
                        zip(feature_list[model.ranking_ == 1],
                            model.estimator_.coef_))
        print model.estimator_
        if hasattr(model.estimator_, 'coef_'):
            print model.estimator_.coef_

def debug_plots(model, ytrue, ypred, prefix):
    """ scatter plot of ytrue/ypred, plot of residuals """
    import matplotlib
    matplotlib.use('Agg')
    import pylab as pl

    pl.clf()
    ymax = max(ytrue.max(), ypred.max())
    pl.plot([0, ymax], [0, ymax], 'k--', lw=4)
    pl.scatter(ytrue, ypred)
    pl.savefig('%s.png' % prefix)

    pl.clf()
    errs = ytrue-ypred
    hmax = max(abs(errs.min()), abs(errs.max()))
    xbins = np.linspace(-hmax, hmax, 20)
    pl.hist(errs, bins=xbins, histtype='step', normed=True)
    pl.savefig('%s_errors.png' % prefix)

def test_model(model, xtrain, ytrain, feature_list, prefix):
    """ use train_test_split to create validation train/test samples """
    xTrain, xTest, yTrain, yTest = train_test_split(xtrain, ytrain,
                                                    test_size=0.4)

    if DO_RFECV:
        model.fit(xtrain, ytrain)
        if hasattr(model, 'coef_'):
            model = RFECV(estimator=model, verbose=0, step=1,
                          scoring=score_fn, cv=3)

    model.fit(xTrain, yTrain)
    print 'score', model.score(xTest, yTest)
    ypred = model.predict(xTest)
    ### don't allow model to predict negative number of orders
    if any(ypred < 0):
        print ypred[ypred < 0]
        ypred[ypred < 0] = 0

    print 'RMSE', np.sqrt(mean_squared_error(ypred, yTest))

#    debug_output(model, feature_list)

    debug_plots(model, yTest, ypred, prefix)

    return

def write_prediction(model, xtrain, ytrain, xtest, ytest, prefix):
    """ refit model with full training set, write prediction to file """
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    
    if any(ypred < 0):
        print ypred[ypred < 0]
        ypred[ypred < 0] = 0

    ytest['Prediction'] = ypred

    ytest.to_csv('submit_%s.csv' % prefix, index=False, float_format='%.2f')

    return

def my_model(xtrain, ytrain, xtest, ytest, feature_list):
    """ compare Linear Regression to RandomForest Regressor """
    model = LinearRegression(fit_intercept=True, normalize=True)
    test_model(model, xtrain, ytrain, feature_list,
                prefix=repr(model).split('(')[0])
    write_prediction(model, xtrain, ytrain, xtest, ytest, 
                     prefix='linear_regression')

    model = RandomForestRegressor(n_jobs=-1, n_estimators=40)
    test_model(model, xtrain, ytrain, feature_list,
                prefix=repr(model).split('(')[0])
    write_prediction(model, xtrain, ytrain, xtest, ytest, 
                     prefix='random_forest')

    return

if __name__ == '__main__':
    xtrain, ytrain, xtest, ytest, feature_list = load_data()

    my_model(xtrain, ytrain, xtest, ytest, feature_list)
