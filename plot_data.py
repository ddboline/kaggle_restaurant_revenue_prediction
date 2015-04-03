#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Module to plot data, create simple html page

@author: Daniel Boline <ddboline@gmail.com>
"""

import os

import matplotlib
matplotlib.use('Agg')
import pylab as pl

import numpy as np
#from pandas.tools.plotting import scatter_matrix

def create_html_page_of_plots(list_of_plots, prefix='html'):
    """
    create html page with png files
    """
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    os.system('mv *.png %s' % prefix)
    #print(list_of_plots)
    idx = 0
    htmlfile = open('%s/index_0.html' % prefix, 'w')
    htmlfile.write('<!DOCTYPE html><html><body><div>\n')
    for plot in list_of_plots:
        if idx > 0 and idx % 200 == 0:
            htmlfile.write('</div></html></html>\n')
            htmlfile.close()
            htmlfile = open('%s/index_%d.html' % (prefix, (idx//200)), 'w')
            htmlfile.write('<!DOCTYPE html><html><body><div>\n')
        htmlfile.write('<p><img src="%s"></p>\n' % plot)
        idx += 1
    htmlfile.write('</div></html></html>\n')
    htmlfile.close()

### Specify histogram binning by hand
BOUNDS = {'Id': [0, 10000, 50],
            'Open Date': [0, 7000, 50],
            'City Group': [0, 2, 2],
            'Type': [0, 4, 4],
            'P1': [1, 16, 16],
            'P2': [1, 8, 8],
            'P3': [0, 8, 9],
            'P4': [2, 8, 7],
            'P5': [1, 9, 9],
            'P6': [1, 11, 11],
            'P7': [1, 11, 11],
            'P8': [1, 11, 11],
            'P9': [4, 11, 7],
            'P10': [4, 11, 7],
            'P11': [1, 11, 11],
            'P12': [2, 11, 10],
            'P13': [3, 8, 5],
            'P14': [0, 16, 16],
            'P15': [0, 11, 11],
            'P16': [0, 16, 16],
            'P17': [0, 16, 16],
            'P18': [0, 16, 16],
            'P19': [1, 26, 25],
            'P20': [1, 16, 16],
            'P21': [1, 16, 16],
            'P22': [1, 6, 6],
            'P23': [1, 26, 26],
            'P24': [0, 11, 11],
            'P25': [0, 11, 11],
            'P26': [0, 13, 13],
            'P27': [0, 13, 13],
            'P28': [1, 13, 12],
            'P29': [0, 11, 11],
            'P30': [0, 26, 26],
            'P31': [0, 16, 16],
            'P32': [0, 26, 26],
            'P33': [0, 7, 7],
            'P34': [0, 31, 31],
            'P35': [0, 16, 16],
            'P36': [0, 21, 21],
            'P37': [0, 9, 9],
            'Type0': [0, 2, 2],
            'Type1': [0, 2, 2],
            'Type2': [0, 2, 2],
            'Type3': [0, 2, 2],
            'revenue': [0, 20e6, 50],}

def plot_data(indf, prefix='html'):
    """
    create scatter matrix plot, histograms
    """
    list_of_plots = []
#    scatter_matrix(indf)
#    pl.savefig('scatter_matrix.png')
#    list_of_plots.append('scatter_matrix.png')

    for col in indf:
        pl.clf()
#        v = indf[col]
#        nent = len(v)
#        hmin, hmax = v.min(), v.max()
#        xbins = np.linspace(hmin,hmax,nent)
        hmin, hmax, nbin = BOUNDS[col]
        xbins = np.linspace(hmin, hmax, nbin)
        indf[col].hist(bins=xbins, histtype='step', normed=True)
        pl.title(col)
        pl.savefig('%s_hist.png' % col)
        list_of_plots.append('%s_hist.png' % col)

    create_html_page_of_plots(list_of_plots, prefix)
    return
