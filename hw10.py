import pickle

import numpy as np
import math

import pandas as pd
from scipy import ndimage
import random
from collections import Counter


import plotly.graph_objs as go
import plotly.express as px
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots


def main():
    print(f"Started generating forests at {pd.Timestamp.now()}...")
    for L in [64, 128]:
        for D in [1, 2, L, L*L]:
            print(f"Starting L={L}, D={D} at {pd.Timestamp.now()}...")
            forest = make_forest(L, D)
            file_path = f"./Data/Forest, L={L}, D={D}"
            with open(file_path, 'wb') as file:
                pickle.dump(forest, file)
            print(f"Forest fully forested.")
    print(f"Finished generating forests at {pd.Timestamp.now()}.")


def get_prob_grid(L):
    p_sum = sum(
        [spark_probability(i+1, j+1, L) for i in range(L) for j in range(L)]
    )
    forest = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            forest[i][j] = spark_probability(i+1, j+1, L) / p_sum
    return forest


def spark_probability(i, j, L):
    l = L/10.0
    prob = math.exp(-i/l)*math.exp(-j/l)
    return prob


def make_forest(L, D):

    p_sum = sum(
        [spark_probability(i + 1, j + 1, L) for i in range(L) for j in range(L)]
    )

    forest = np.zeros((L, L))
    forests = list()
    forests.append(forest.copy())

    # Add trees until we fill the grid
    for _ in range(L * L):
        # Identify tree free locations on grid
        no_trees = list(zip(*np.where(forest == 0)))

        # List to store yields associated with possible new tree location
        tree_yields = list()
        # List to store tree locations associated with above yields
        tree_locs = list()

        # Assess D different possible tree placements
        for _ in range(D):
            # Find a random empty spot where you could plant a tree
            tree_i, tree_j = random.choice(no_trees)

            # Store location we're assessing
            tree_locs.append((tree_i, tree_j))
            # Plant the tree temporarily, and get the expected yield.
            forest[tree_i][tree_j] = 1
            labeled, num_clusters = ndimage.measurements.label(forest)
            forest_size = np.sum(forest)

            # List to store yields associated with specific sparks for the
            # tree location we're assessing.
            yields = list()
            for i in range(L):
                for j in range(L):
                    spark_prob = spark_probability(i + 1, j + 1, L) / p_sum
                    if forest[i][j] == 0:
                        yields.append(spark_prob)
                        continue
                    cluster_size = len(np.where(labeled == labeled[i][j])[0])
                    yields.append(
                        spark_prob *
                        ((forest_size - cluster_size) / forest_size)
                    )
            tree_yields.append(np.mean(yields))

            # Yank tree
            forest[tree_i, tree_j] = 0

        # Find index of highest yield and plant at corresponding location
        tree_i, tree_j = tree_locs[tree_yields.index(max(tree_yields))]
        forest[tree_i, tree_j] = 1
        forests.append(forest.copy())

    return forests


def analyze():
    files = [r'/Users/janeyalex/Documents/Pocs/HW10/data/Forest, L=32, D=1',
             r'/Users/janeyalex/Documents/Pocs/HW10/data/Forest, L=32, D=2',
             r'/Users/janeyalex/Documents/Pocs/HW10/data/Forest, L=32, D=32',
             r'/Users/janeyalex/Documents/Pocs/HW10/data/Forest, L=32, D=1024']

    forest_dict = {}

    for file in files:
        f = open(file, 'rb')
        forests = []
        while True:
            try:
                forests.append(pickle.load(f))
            except EOFError:
                break

        forest_dict[file.split('=')[-1]] = forests

    p_sum = sum(
        [spark_probability(i + 1, j + 1, 32) for i in range(32) for j in range(32)]
    )

    yields_dict = {}
    best_forests_dict = {}
    for D in forest_dict.keys():
        forests_with_D_tests = forest_dict[D][0]

        tree_yields = []
        for f, forest_ in enumerate(forests_with_D_tests[1:]):
            yields = []

            labeled, num_clusters = ndimage.measurements.label(forest_)
            forest_size = np.sum(forest_)

            for i in range(32):
                for j in range(32):
                    spark_prob = spark_probability(i + 1, j + 1, 32) / p_sum
                    if forest_[i][j] == 0:
                        yields.append(spark_prob)
                        continue
                    cluster_size = len(np.where(labeled == labeled[i][j])[0])
                    y = spark_prob * (forest_size - cluster_size)
                    yields.append(y)

            tree_yields.append(np.mean(yields))

        max_yield_index = tree_yields.index(max(tree_yields))

        yields_dict[D] = tree_yields
        best_forests_dict[D] = forests_with_D_tests[max_yield_index]


        plt.imshow(forests_with_D_tests[max_yield_index])
        plt.title(f'Max Yield for D={D}')
        plt.show()

    # 3b, yield curves
    fig = go.Figure()
    for d_yields in yields_dict.keys():
        to_plot = yields_dict[d_yields]
        steps = [(i+1)/(32**2) for i in range(len(to_plot))]

        fig.add_trace(go.Scatter(x=steps, y=to_plot, name=f'D={d_yields}', mode='lines'))

    fig.update_layout(title=f'Yield Curve for L=32')
    fig.show()

    # 3c zipf for each forest
    fig = go.Figure()
    for best_yield in best_forests_dict.keys():
        to_plot = best_forests_dict[best_yield]
        labeled, num_clusters = ndimage.measurements.label(to_plot)
        ordred_clusters = ndimage.measurements.sum(to_plot, labeled, index=np.arange(labeled.max() + 1))[1:]
        data = dict(Counter(ordred_clusters))
        data = sorted(data.items())
        x, y = zip(*data)

        x = np.log10(np.asarray(x))
        y = np.log10(np.asarray(y))
        fig.add_trace(go.Scatter(x=x, y=y, name=f'D={best_yield}', mode='lines'))

    fig.update_layout(title=f'Distributions of Tree Component sizes S at Peak Yield')
    fig.show()

    # 3d
    fig = go.Figure()
    data = forest_dict['1024'][0]
    denominator = 32**2
    for step in data:
        if np.sum(step) % 100 == 0 and np.sum(step) != 0:

            labeled, num_clusters = ndimage.measurements.label(step)
            ordred_clusters = ndimage.measurements.sum(step, labeled, index=np.arange(labeled.max() + 1))[1:]
            data = dict(Counter(ordred_clusters))
            data = sorted(data.items())
            x, y = zip(*data)

            x = np.log10(np.asarray(x))
            y = np.log10(np.asarray(y))
            fig.add_trace(go.Scatter(x=x, y=y, name=f'rho={round(np.sum(step) / denominator, 1)}', mode='lines'))

    fig.update_layout(title=f'Distributions of Tree Component sizes for D=1024')
    fig.show()


if __name__ == '__main__':
    analyze()
