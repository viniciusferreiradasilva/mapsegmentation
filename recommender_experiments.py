import argparse
import pandas as pd
import numpy as np
import inspect

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

# Required recommendation algorithm argument.
parser.add_argument('--recommender_algorithm', type=int, required=True,
                    help='An integer representing the clustering algorithm that will be used to segment the map:\n'
                         '0 - Nearest District. Args: attribute_name (str), threshold (float),\n'
                         '1 - Nearest District for all categories. Args: n_clusters (int),\n')

# Required clustering algorithm argument.
parser.add_argument('--clustering_algorithm', type=int, required=True,
                    help='An integer representing the clustering algorithm that will be used to segment the map:\n'
                         '0 - Highest attribute. Args: attribute_name (str), threshold (float),\n'
                         '1 - K-means. Args: n_clusters (int),\n'
                         '2 - DBSCAN. Args eps (float), min_samples (int),\n'
                         '3 - Agglomerative clustering. Args: n_clusters (int)\n')

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

# List of clustering algorithms.
clustering_algorithms = [highest_attribute_value, kmeans, dbscan, agglomerative_clustering]
# List of recommender algorithms.
recommender_algorithms = [nearest_district, nearest_district_for_all_categories]
# Removes na values.
df = df.fillna({'categories': ''})
# The results of precision.
results = np.empty(len(configs))
# Run the clustering and the recommendation for each config and retrieve the precision values.

# Retrieve all the categories in the dataset.
categories = list(set(','.join([df.at[x, 'categories'].replace(' ', '') for x in range(len(df))]).split(',')))
df_training, df_test = train_test_split(df, test_size=0.5)
df_training = df_training.reset_index()
df_test = df_test.reset_index()

for config_index, config in enumerate(configs):
    print((config_index + 1), ' of ', len(configs))
    # Clustering the full dataset to get the ground-truth values of cluster.
    df = clustering_algorithms[args.clustering_algorithm](df, *config)
    # Clustering the training dataset.
    df_training = clustering_algorithms[args.clustering_algorithm](df_training, *config)
    # Groups the dataframe by the clustering id, generating a dataframe with two columns: one representing the cluster
    # id and the other representing the list of categories. This command returns a Pandas Series.
    districts = df_training.groupby(by='cluster_id', as_index=False)['categories'] \
        .apply(lambda x: '{}'.format(','.join(x)))
    # Converting the pandas Series to DataFrame.
    districts = pd.DataFrame({'cluster_id': districts.index, 'categories': districts.values})
    # Builds a dictionary where the key represents the category and the value represents the node id of the category.
    categories = dict(
        zip(categories, range(len(districts['cluster_id']), len(categories) + len(districts['cluster_id']))))
    # Builds a list of dicts where one represents the categories and the number of appears of this category.
    districts_categories_count = [None] * len(districts)

    for index, row in districts.iterrows():
        counter = Counter(row[0].replace(' ', '').split(','))
        districts_categories_count[row[1]] = dict(zip([categories[x] for x in counter.keys()], counter.values()))
    # Calculates the embeddings of the districts.
    districts_embedding = embedding_by_category_probability(categories.values(), districts_categories_count)

    # Builds a list of dicts where one represents the categories and the number of appears of this category.
    test_categories_count = [None] * len(df_test)
    for index, row in enumerate(df_test['categories']):
        counter = Counter(row.replace(' ', '').split(','))
        test_categories_count[index] = dict(zip([categories[x] for x in counter.keys()], counter.values()))

    # Calculates the embeddings for the training set.
    test_embedding = embedding_by_category_probability(categories.values(), test_categories_count)
    predicted_districts = recommender_algorithms[args.recommender_algorithm](df_training, df_test, categories,
                                                                             districts_embedding, test_embedding)
    # Retrieves the test values.
    y_test = list(df.loc[df['business_id'].isin(list(df_test['business_id']))]['cluster_id'])
    y_predicted = predicted_districts
    # Calculates the classification metrics.
    # results[config_index] = accuracy_score(y_test, y_predicted, average='weighted', labels=np.unique(y_predicted))
    results[config_index] = accuracy_score(y_test, y_predicted)

recommender_file = args.output_dir + args.input_file.split('/')[-1].split('.')[0] + '_' + clustering_algorithms[
        args.clustering_algorithm].__name__ + '_' + recommender_algorithms[args.recommender_algorithm].__name__ + \
                   '_recommendation.csv'
print("saving recommender analysis into:", recommender_file)
# Writing the embedding into a file.
with open(recommender_file, 'w') as f:
    # Writes the header.
    f.write(','.join(inspect.getfullargspec(clustering_algorithms[args.clustering_algorithm])[0][1:])
            + ',' + 'result' + '\n')
    for index, result in enumerate(results):
        # Writes the lines.
        f.write(','.join(map(str, configs[index])) + ',' + str(result) + '\n')
f.close()
