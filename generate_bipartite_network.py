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
from analysis.map_drawing import plot_map
from analysis.metrics import silhouette_coefficient
from analysis.metrics import silhouette_sample
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
                         '1 - Highest attribute. Args: attribute_name (str), threshold (float),\n'
                         '2 - K-means. Args: n_clusters (int),\n'
                         '3 - DBSCAN. Args eps (float), min_samples (int),\n'
                         '4 - Agglomerative clustering. Args: n_clusters (int)\n')

# Required clustering algorithm argument.
parser.add_argument('--args', required=False, action='append', default=[],
                    help='List of arguments that will be passed to the clustering algorithm. Each clustering algorithm '
                         'has a certain number of parameters. You should pass them in the same order:\n'
                         '--args ARG_1 --args ARG_2 ... --args ARG_N.')

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
# Builds a list of dicts where one represents the categories and the number of appears of this category.
districts_categories_count = [None] * len(districts)
for index, row in districts.iterrows():
    counter = Counter(row['categories'].replace(' ', '').split(','))
    districts_categories_count[row['cluster_id']] = dict(zip([categories[x] for x in counter.keys()], counter.values()))

# Retrieve the list of nodes used to build the network. The vertices are the union between districts and categories ids.
vertices_list = list(districts['cluster_id']) + list(categories.values())
# Retrieve the list of edges used to build the network. The edges are tuples of districts and categories ids.
edge_list = [(x, y) for x in list(districts['cluster_id']) for y in districts_categories_count[x]]
# Retrieve the list of edge weights to build the network.
weights = list(itertools.chain.from_iterable([list(x.values()) for x in districts_categories_count]))
# Creates the network.
g = create_network(len(vertices_list))
g = add_weighted_edges(g, edge_list, weights)

number_of_districts = len(districts)
number_of_categories = len(categories)

fig = draw_pointed_cluster_map(df)
# Saving the map on the dir.
map_output_file = args.output_dir + args.input_file.split('/')[-1].split('.')[0] + '_' +\
                  clustering_algorithms[args.clustering_algorithm].__name__ + '.eps'
print("saving map into:", map_output_file)
save_map(fig, map_output_file)

# Saving the network as .ncol on the dir.
network_output_file = args.output_dir + args.input_file.split('/')[-1].split('.')[0] + '_' + clustering_algorithms[
    args.clustering_algorithm].__name__ + '_' + str(number_of_districts) + '_' + str(number_of_categories) + '.ncol'
print("saving network district - category into:", network_output_file)
g.write_ncol(f=network_output_file, names='', weights='weight')

districts_embedding = embedding_by_category_probability(categories, districts_categories_count)
embedding_file = args.output_dir + args.input_file.split('/')[-1].split('.')[0] + '_' + clustering_algorithms[
        args.clustering_algorithm].__name__ + '_' + str(number_of_districts) + '_' + str(number_of_categories) +\
                 '_district_emb.txt'
print("saving embedding from districts into:", embedding_file)
# Writing the embedding into a file.
with open(embedding_file, 'w') as f:
    for district_embedding in districts_embedding:
        f.write(' '.join(["{0:.10f}".format(x) for x in list(district_embedding)]) + "\n")
f.close()

districts_embedding = np.transpose(embedding_by_category_probability(categories, districts_categories_count))
embedding_file = args.output_dir + args.input_file.split('/')[-1].split('.')[0] + '_' + clustering_algorithms[
        args.clustering_algorithm].__name__ + '_' + str(number_of_districts) + '_' + str(number_of_categories) +\
                 '_categories_emb.txt'
print("saving embedding from categories into:", embedding_file)
# Writing the embedding into a file.
with open(embedding_file, 'w') as f:
    for district_embedding in districts_embedding:
        f.write(' '.join(["{0:.10f}".format(x) for x in list(district_embedding)]) + "\n")
f.close()

print('-' * 10)
