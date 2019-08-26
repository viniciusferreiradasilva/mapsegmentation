import argparse
import pandas as pd
import numpy as np
import itertools

from preprocessing.file_loading import json_2_dataframe
from preprocessing.file_loading import load_attribute
from preprocessing.network_processing import create_network
from preprocessing.network_processing import add_weighted_edges
from algorithms.clustering import highest_attribute_value
from algorithms.clustering import kmeans
from algorithms.clustering import dbscan
from algorithms.clustering import agglomerative_clustering
from algorithms.embedding import embedding_by_category_probability
from analysis.map_drawing import draw_pointed_cluster_map
from analysis.map_drawing import save_map
from collections import Counter

# Instantiate the parser
parser = argparse.ArgumentParser(description='Map segmentation tool.', formatter_class=argparse.RawTextHelpFormatter)

# Required input_file name argument.
parser.add_argument('--input_file', required=True, type=str,
                    help='A string representing a .json input file path with latitude and longitude columns.')

# Optional output_file dir argument.
parser.add_argument('--output_dir', type=str, help="A string representing an output dir path to save the clustering"
                                                   " map, the resulting network .ncol file and the resulting embeddings"
                    , required=True)

# Required clustering algorithm argument.
parser.add_argument('--clustering_algorithm', type=int, required=True,
                    help='An integer representing the clustering algorithm that will be used to segment the map:\n'
                         '0 - Highest attribute. Args: attribute_name (str), threshold (float),\n'
                         '1 - K-means. Args: n_clusters (int),\n'
                         '2 - DBSCAN. Args eps (float), min_samples (int),\n'
                         '3 - Agglomerative clustering. Args: n_clusters (int)\n')

# Required clustering algorithm argument.
parser.add_argument('--args', required=False, nargs='+', default=[],
                    help='List of arguments that will be passed to the clustering algorithm. Each clustering algorithm '
                         'has a certain number of parameters. You should pass them in the same order:\n'
                         '--args ARG_1 ARG_2 ... ARG_N.')

args = parser.parse_args()
# Loading a json file for a pandas dataframe.
print("loading data " + args.input_file)
df = json_2_dataframe(args.input_file)
# Loading check-in list to the dataframe.
df = load_attribute(df, 'data/input/filtered/checkins.json', 'business_id', 'checkins')

print("clustering...")
clustering_algorithms = [highest_attribute_value, kmeans, dbscan, agglomerative_clustering]
df = clustering_algorithms[args.clustering_algorithm](df, *args.args)

# Groups the dataframe by the clustering id, generating a dataframe with two columns: one representing the cluster
# id and the other representing the list of categories. This command returns a Pandas Series.
districts = df.fillna({'categories': ''}).groupby(by='cluster_id', as_index=False)['categories']\
    .apply(lambda x: '{}'.format(','.join(x)))
# Converting the pandas Series to DataFrame.
districts = pd.DataFrame({'cluster_id': districts.index, 'categories': districts.values})
# Lists all the categories of all districts.
categories = list(dict.fromkeys((districts['categories'].str.cat(sep=',').replace(' ', '').split(','))))
# Builds a dictionary where the key represents the category and the value represents the node id of the category.
categories = dict(zip(categories, range(len(districts['cluster_id']), len(categories) + len(districts['cluster_id']))))
csv_file = args.output_dir + args.input_file.split('/')[-1].split('.')[0] + '_' + clustering_algorithms[
        args.clustering_algorithm].__name__ + '_districts.csv'
f = open(csv_file, 'w')
f.write('cluster_id,' + ','.join(categories) + '\n')
for index, row in districts.iterrows():
    district_categories = np.array([row['categories'].replace(' ', '').split(',').count(category) for category in categories])
    district_categories = district_categories / np.max(district_categories)
    f.write(str(row['cluster_id'])+ ',' + ','.join([str(x) for x in district_categories]) + '\n')
f.close()
