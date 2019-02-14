#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np

# Draw a simple map with matplotlib.
def draw_cluster_map(df):
    fig = plt.figure()
    plt.scatter(df.latitude, df.longitude, marker='.', c=df.cluster_id)
    return fig


# Draw a simple map with matplotlib.
def draw_pointed_cluster_map(df, attribute='checkins'):
    fig = plt.figure()
    s = list(df[attribute]/(np.mean(df[attribute])))
    plt.scatter(df.latitude, df.longitude, marker='.', s=s, c=df.cluster_id)
    return fig


# Saves the map into a dir.
def save_map(fig, file_name):
    fig.savefig(file_name, format='eps')
    plt.close(fig)


# Plots the map.
def plot_map(fig):
    plt.show(fig)


