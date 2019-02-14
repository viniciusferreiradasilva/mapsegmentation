import argparse
import pandas as pd
import itertools

from preprocessing.pre_processing import json_2_dataframe
from preprocessing.pre_processing import load_attribute
from preprocessing.network_processing import create_network
from preprocessing.network_processing import add_weighted_edges
from algorithms.clustering import highest_attribute_value
from algorithms.clustering import kmeans
from algorithms.clustering import dbscan
from algorithms.clustering import agglomerative_clustering
from mapdrawing.map_drawing import draw_pointed_cluster_map
from mapdrawing.map_drawing import save_map
from mapdrawing.map_drawing import plot_map
from collections import Counter

# Instantiate the parser
parser = argparse.ArgumentParser(description='Map segmentation tool.')

# Required input_file name argument.
parser.add_argument('--input_file', required=True, type=str,
                    help='An input .json file with latitude and longitude columns.')

# Optional output_file dir argument.
parser.add_argument('--output_dir', type=str, help='An output dir path to save the clustering map.')

# Required clustering algorithm argument.
parser.add_argument('--clustering_algorithm', type=int, required=False,
                    help='Clustering algorithm that will be used to segment the map: (1 - Highest attribute, '
                         '2 - K-means, 3 - DBSCAN, 4 - Agglomerative clustering)')

args = parser.parse_args()

# Loading a json file for a pandas dataframe.
print("loading data " + args.input_file)
df = json_2_dataframe(args.input_file)
# Loading check-in list to the dataframe.
df = load_attribute(df, 'data/input/filtered/checkins.json', 'business_id', 'checkins')

print("clustering...")
clustering_algorithms = [highest_attribute_value, kmeans, dbscan, agglomerative_clustering]
df = clustering_algorithms[args.clustering_algorithm](df)


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
    counter = Counter(row[0].replace(' ', '').split(','))
    districts_categories_count[row[1]] = dict(zip([categories[x] for x in counter.keys()], counter.values()))


# Retrieve the list of nodes used to build the network. The vertices are the union between districts and categories ids.
vertices_list = vertices_list = list(districts['cluster_id']) + list(categories.values())
# Retrieve the list of edges used to build the network. The edges are tuples of districts and categories ids.
edge_list = [(x, y) for x in list(districts['cluster_id']) for y in districts_categories_count[x]]
# Retrieve the list of edge weights to build the network.
weights = list(itertools.chain.from_iterable([list(x.values()) for x in districts_categories_count]))
# Creates the network.
g = create_network(len(vertices_list))
g = add_weighted_edges(g, edge_list, weights)


print("drawing map...")
fig = draw_pointed_cluster_map(df)
# If an output dir is passed, the resulting map is saved into this dir.
if args.output_dir:
    # Saving the map on the dir.
    map_output_file = args.output_dir + args.input_file.split('/')[-1].split('.')[0] + '_' +\
                      clustering_algorithms[args.clustering_algorithm].__name__ + '.eps'
    print("saving map into:", map_output_file)
    save_map(fig, map_output_file)
    # Saving the network as .ncol on the dir.
    network_output_file = args.output_dir + args.input_file.split('/')[-1].split('.')[0] + '_' + clustering_algorithms[
        args.clustering_algorithm].__name__ + '_' + str(g.vcount()) + '_' + str(g.ecount()) + '.ncol'
    print("saving network district - category into:", network_output_file)
    g.write_ncol(f=network_output_file, names='', weights='weight')
else:
    plot_map(fig)
