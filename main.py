from preprocessing.pre_processing import json_2_dataframe
from preprocessing.clustering_processing import clustering_by_highest_attribute_value
from preprocessing.clustering_processing import clustering_by_kmeans
from preprocessing.clustering_processing import clustering_by_dbscan
from preprocessing.clustering_processing import clustering_by_agglomerative_clustering
from mapdrawing.map_drawing import draw_cluster_map

# filename = 'data/raw/business.json'
# filename = 'data/cities/Pittsburgh.json'
# filename = 'data/cities/Las Vegas.json'
# filename = 'data/cities/Phoenix.json'
# filename = 'data/cities/test.json'
filename = 'data/cities/Charlotte.json'


# Loading a json file from a pandas dataframe.
print("loading data...")
df = json_2_dataframe(filename)

print("clustering...")
# df = clustering_by_highest_attribute_value(df, 'review_count', 10)
# df = clustering_by_kmeans(df, 200)
# df = clustering_by_dbscan(df)
df = clustering_by_agglomerative_clustering(df, n_clusters=100)

print("drawing map...")
draw_cluster_map(df)
