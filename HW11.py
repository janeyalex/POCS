#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 12:14:42 2021

@author: janeyalex
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import math
import networkx as nx


def func(k):
    return 3.46 * 10**8 * k**(-0.661)


def q1():
    y = []
    for i in range(1, 200):
        y.append(func(i))

    x = [i for i in range(1,200)]
    x = np.asarray(x)
    y = np.asarray(y)
    x_log = np.log10(x)
    y_log = np.log10(y)

    #fit_x = linregress(x, y)
    fit_x_log = linregress(x_log, y_log)


    mean = sum(x*y)/sum(y)
    numer = sum((np.array(x, dtype=float)**2)*y)
    stdv = np.sqrt(((numer/sum(y)) - mean**2))

    print('mean: ' + str(mean) + " | numer: " + str(numer) + ' | stdv: '+ str(stdv))

    plt.scatter(x_log, y_log)
    plt.title(f'Slope: {fit_x_log.slope:.3f}  | Intercept:  {fit_x_log.intercept:.3f}')
    plt.xlabel('Log10 k (frequency)')
    plt.ylabel('Log10 N sub k (number of words that appear k times)')
    plt.show()
    
    unlogged_words_once = func(1)
    sum_words = 0
    for i in range(1, 10**7+1):
        sum_words += i*func(i)

    frac_words_that_appear_once = unlogged_words_once / sum_words

    print(f'Fraction of words that appear one time: {frac_words_that_appear_once}')
    # # c ii
    total_unique_words = 0
    for i in range(1, 10**7+1):
        total_unique_words += func(i)

    print(f'Amount of words that appear one time: {unlogged_words_once}')
    print(f'Proportion of words that appear once time: {unlogged_words_once/total_unique_words}')

    total_left_out = 0
    for i in range(1, 200):
        total_left_out += i*y[i-1]

    print(f'Proportion of words left out by Google: {total_left_out/sum_words}')


def q2():
    probs = list(np.logspace(-4,0,14))
    CpList = np.zeros((10, 14))
    LpList = np.zeros((10, 14))

    Gzero = nx.watts_strogatz_graph(1000, 10, 0, seed=None)
    C0 = nx.average_clustering(Gzero)
    L0 = nx.average_shortest_path_length(Gzero)

    for i in range(10):
        curC0 = []
        curL0 = []
        for prob in probs:
            G = nx.watts_strogatz_graph(1000, 10, prob, seed=None)
            Cp = nx.average_clustering(G)
            Lp = nx.average_shortest_path_length(G)

            curC0.append(Cp/C0)
            curL0.append(Lp/L0) 

        CpList[i, :] = np.asarray(curC0)
        LpList[i, :] = np.asarray(curL0)

    CpMeans = np.mean(CpList, axis=0)
    LpMeans = np.mean(LpList, axis=0) 

    
    plt.scatter(probs,CpMeans,label='C(p)/C(0)')
    plt.scatter(probs,LpMeans,label='L(p)/L(0)')
    plt.xscale('log')
    plt.xlabel('p')
    plt.title('Clustering and Average Shortest Path for Different Proababilities in a Graph')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    q1()
    q2()

    
