import argparse
import pandas as pd
import numpy as np

from preprocessing.file_loading import json_2_dataframe
from preprocessing.file_loading import load_attribute
from algorithms.clustering import highest_attribute_value
from algorithms.clustering import kmeans
from algorithms.clustering import dbscan
from algorithms.clustering import agglomerative_clustering
from algorithms.embedding import embedding_by_category_probability
from algorithms.recommender import nearest_district_for_all_categories
from algorithms.recommender import nearest_district
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score

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

# Removes na values.
df = df.fillna({'categories': ''})

# Groups the dataframe by the clustering id, generating a dataframe with two columns: one representing the cluster
# id and the other representing the list of categories. This command returns a Pandas Series.
districts = df.groupby(by='cluster_id', as_index=False)['categories']\
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

# Calculates the embeddings of the districts.
districts_embedding = embedding_by_category_probability(categories, districts_categories_count)

# Splits the original dataset into training and test set.
df_training, df_test = train_test_split(df, test_size=0.5)
# Builds a list of dicts where one represents the categories and the number of appears of this category.

training_categories_count = [None] * len(df_training)
for index, row in enumerate(df_training['categories']):
    counter = Counter(row.replace(' ', '').split(','))
    training_categories_count[index] = dict(zip([categories[x] for x in counter.keys()], counter.values()))

# Builds a list of dicts where one represents the categories and the number of appears of this category.
test_categories_count = [None] * len(df_test)
for index, row in enumerate(df_test['categories']):
    counter = Counter(row.replace(' ', '').split(','))
    test_categories_count[index] = dict(zip([categories[x] for x in counter.keys()], counter.values()))

# Calculates the embeddings for the training and test.
training_embedding = embedding_by_category_probability(categories, training_categories_count)
test_embedding = embedding_by_category_probability(categories, test_categories_count)

# Does the recommendation.
predicted_districts = nearest_district_for_all_categories(df_training, df_test, categories, training_embedding, test_embedding)
y_test = list(df_test['cluster_id'])
y_predicted = predicted_districts
# Calculates the classification metrics.
print("accuracy_score:", accuracy_score(y_test, y_predicted))
print("precision_score:", precision_score(y_test, y_predicted, average='weighted', labels=np.unique(y_predicted)))

print("-" * 50)

# Does the recommendation.
predicted_districts = nearest_district(df_training, df_test, categories, training_embedding, test_embedding)
y_test = list(df_test['cluster_id'])
y_predicted = predicted_districts
# Calculates the classification metrics.
print("accuracy_score:", accuracy_score(y_test, y_predicted))
print("precision_score:", precision_score(y_test, y_predicted, average='weighted', labels=np.unique(y_predicted)))
