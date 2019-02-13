import argparse
import pandas as pd

from preprocessing.pre_processing import json_2_dataframe
from preprocessing.pre_processing import load_attribute
from algorithms.clustering import highest_attribute_value
from algorithms.clustering import kmeans
from algorithms.clustering import dbscan
from algorithms.clustering import agglomerative_clustering
from mapdrawing.map_drawing import draw_pointed_cluster_map
from mapdrawing.map_drawing import save_map
from mapdrawing.map_drawing import plot_map

# filename = 'data/input/raw/business.json'
# filename = 'data/input/cities/Pittsburgh.json'
# filename = 'data/input/cities/Las Vegas.json'
# filename = 'data/input/cities/Phoenix.json'
# filename = 'data/input/cities/test.json'
# filename = 'data/input/cities/Charlotte.json'

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
print("loading data...")
df = json_2_dataframe(args.input_file)
# Loading checkins to the dataframe.
df = load_attribute(df, 'data/input/filtered/checkins.json', 'business_id', 'checkins')

print("clustering...")
# clustering_algorithms = [highest_attribute_value, kmeans, dbscan, agglomerative_clustering]
# df = highest_attribute_value(df, 'checkins', 10)
df = kmeans(df, 200)
# df = dbscan(df)
# df = agglomerative_clustering(df, n_clusters=100)

# Groups the dataframe by the clustering id, generating a dataframe with two columns: one representing the cluster
# id and the other representing the list of categories. This command returns a Pandas Series.
districts = df.fillna({'categories': ''}).groupby(by='cluster_id', as_index=False)['categories']\
    .apply(lambda x: '{}'.format(','.join(x)))
# Converting the pandas Series to DataFrame.
districts = pd.DataFrame({'cluster_id': districts.index, 'categories': districts.values})
# print(districts.head())

print("drawing map...")
fig = draw_pointed_cluster_map(df)
# If an output dir is passed, the resulting map is saved into this dir.
if args.output_dir:
    output_file = args.output_dir + args.input_file.split('/')[-1].split('.')[0] + '.eps'
    print("saving into:", output_file)
    save_map(fig, output_file)
else:
    plot_map(fig)
