from __future__ import print_function
from sklearn.metrics import silhouette_samples, silhouette_score
from collections import Counter
from preprocessing.file_loading import json_2_dataframe
from preprocessing.file_loading import load_attribute
from algorithms.clustering import highest_attribute_value
from algorithms.clustering import kmeans
from algorithms.clustering import dbscan
from algorithms.clustering import agglomerative_clustering
from utils.utils import str2bool

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import argparse


# Instantiate the parser
parser = argparse.ArgumentParser(description='Map segmentation tool.', formatter_class=argparse.RawTextHelpFormatter)

# Required input_file name argument.
parser.add_argument('--input_file', required=True, type=str,
                    help='A string representing a .json input file path with latitude and longitude columns.')

# Optional output_file dir argument.
parser.add_argument('--output_dir', type=str, help="A string representing an output dir path to save the clustering"
                                                   " map, the resulting network .ncol file and the resulting embeddings"
                    , required=True)

# Required input_file name argument.
parser.add_argument('--map', required=True, type=str, default='false',
                    help='A string representing a .json input file path with latitude and longitude columns.')


# Required clustering algorithm argument.
parser.add_argument('--clustering_algorithm', type=int, required=True,
                    help='An integer representing the clustering algorithm that will be used to segment the map:\n'
                         '1 - Highest attribute. Args: attribute_name (str), threshold (float),\n'
                         '2 - K-means. Args: n_clusters (int),\n'
                         '3 - DBSCAN. Args eps (float), min_samples (int),\n'
                         '4 - Agglomerative clustering. Args: n_clusters (int)\n')

# Required clustering algorithm configs.
parser.add_argument('--configs', required=False, nargs='+', default=[],
                    help='List of configs that will be passed to the clustering algorithm. Each clustering algorithm '
                         'has a certain number of parameters. You should pass them in the same order as strings:\n'
                         '--configs "CONF_1", "CONF_2",... "CONF_N".')


args = parser.parse_args()
# Retrieve the configs from arguments.
configs = [list(map(lambda x: float(x) if x.isdigit() else x, x)) for x in list(map((lambda x: x.strip().split(' ')),
                                                                                    args.configs))]

# Loading a json file for a pandas dataframe.
print("loading data " + args.input_file)
df = json_2_dataframe(args.input_file)
# Loading check-in list to the dataframe.
df = load_attribute(df, 'data/input/filtered/checkins.json', 'business_id', 'checkins')
# List containing all the implemented clustering algorithms.
clustering_algorithms = [highest_attribute_value, kmeans, dbscan, agglomerative_clustering]
# Retrieve the matrix formed by latitude and longitude.
X = df[['latitude', 'longitude']].values
# It will be map plotting?
plot_map = str2bool(args.map)
# Var that save all silhouettes.
silhouettes = np.empty(len(configs))

for index, config in enumerate(configs):
    # Clustering
    cluster_labels = np.array(clustering_algorithms[args.clustering_algorithm](df, *config)['cluster_id'])
    n_clusters = len(Counter(cluster_labels).keys())

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    # Tests if the current config is better than the current best.
    silhouettes[index] = silhouette_avg
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    if plot_map:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
                              edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        ax2.set_title("Clustered data.")
        ax2.set_xlabel("Latitude")
        ax2.set_ylabel("Longitude")

        plt.suptitle(("Silhouette analysis for " + clustering_algorithms[args.clustering_algorithm].__name__ +
                      " on " + args.input_file.split('/')[-1].split('.')[0] + " with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

plt.show()
clustering_file = args.output_dir + args.input_file.split('/')[-1].split('.')[0] + '_' + clustering_algorithms[
        args.clustering_algorithm].__name__ + '.txt'
print("saving clustering analysis into:", clustering_file)
# Writing the embedding into a file.
with open(clustering_file, 'w') as f:
    for index, silhouette in enumerate(silhouettes):
        f.write(''.join(map(str, configs[index])) + ' ' + str(silhouette) + '\n')
    f.write("-" * 50 + "\n")
    max_index = int(np.where(silhouettes == np.amax(silhouettes))[0])
    f.write("Max value: " + ''.join(map(str, configs[max_index])) + " -> " + str(silhouettes[max_index]) + "\n")
    min_index = int(np.where(silhouettes == np.amin(silhouettes))[0])
    f.write("Min value: " + ''.join(map(str, configs[min_index])) + " -> " + str(silhouettes[min_index]) + "\n")
    f.write("Avg value: " + str(np.average(silhouettes)) + "\n")
f.close()
