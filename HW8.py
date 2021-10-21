#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 13:00:03 2021

@author: janeyalex
"""
import random
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from scipy import stats
import itertools
import os
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from scipy import ndimage

# Site percolation
def run_percolation(world, probability):
    N1 = world.shape[0]
    N2 = world.shape[1]
    for i in range(N1): #for cell in every row
        for j in range(N2): #and every column
            die = random.uniform(0, 1)
            if die < probability:
              world[(i,j)] = 1
            else:
              world[(i,j)] = 0

    return (world)


def question_4():
    sizes = [20, 50, 100, 200, 500, 1000]
    probs = [i/100 for i in range(100)][1:]
    averages = {}

    for size in sizes:
        print(size)
        current_averages = []
        for prob in probs:
            world = np.zeros((size, size))
            world = run_percolation(world, prob)

            structure = [[0,1,0],[1,1,1],[0,1,0]] #define connection
            label_world, nb_labels = ndimage.label(world,structure) #label clusters
            sizes = ndimage.sum(world, label_world, range(nb_labels + 1))
            mask = sizes >= sizes.max()
            binary_img = mask[label_world]

            whitespace = binary_img.sum()
            total = size**2
            current_averages.append(whitespace / total)

        averages[size] = current_averages


    fig = go.Figure()
    sizes = [20, 50, 100, 200, 500, 1000]
    for size in sizes:
        y = averages[size]
        x = probs

        fig.add_trace(go.Scatter(x=x, y=y, name=f'L={size}', mode='markers'))

    fig.update_xaxes(title='probability')
    fig.update_yaxes(title='average fractional proportion of lattice')
    fig.update_layout(title=f'percolation')
    fig.write_image('fig4.png', width=1000, height=800, format='png')
    
def question_5():

    current_averages = []

    for i in range(100):
        print(i)
        world = np.zeros((1000, 1000))
        world = run_percolation(world, 0.58)

        structure = [[0,1,0],[1,1,1],[0,1,0]] #define connection
        label_world, nb_labels = ndimage.label(world,structure) #label clusters
        sizes = ndimage.sum(world, label_world, range(nb_labels + 1))
        mask = sizes >= sizes.max()
        binary_img = mask[label_world]

        whitespace = binary_img.sum()
        total = 1000**2
        current_averages.append(whitespace / total)


    fig = ff.create_distplot([current_averages], ['Distribution of Proportion of Lattice for L=1000'], show_hist=False)
    fig.show()
    
question_4()
question_5()