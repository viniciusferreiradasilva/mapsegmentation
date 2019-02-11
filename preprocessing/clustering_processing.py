from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from utils.utils import distance_in_kilometers
NOT_IN_A_CLUSTER = -1


# Clustering by ordering the entries by an attribute and using it as a landmark venue. Putting all the venues with
# a distance (latitude and longitude) <= threshold in the same cluster of the landmark venue.
def clustering_by_highest_attribute_value(df, attribute, threshold):
    # Sorting the values of the dataframe by attribute in the descending order.
    df = df.sort_values(attribute, ascending=False)
    # Creating a new column 'cluster_id' that represents the cluster index of that venue,
    # not considering overlapping.
    current_cluster_index = 0
    clusters_indexes = [NOT_IN_A_CLUSTER] * len(df)
    for venue in df.itertuples():
        if clusters_indexes[venue.Index] == NOT_IN_A_CLUSTER:
            # Assigns the cluster id value to the venue.
            clusters_indexes[venue.Index] = current_cluster_index
            for neighbor in df.itertuples():
                if (venue.Index != neighbor.Index) and (distance_in_kilometers(
                        venue.latitude, venue.longitude, neighbor.latitude, neighbor.longitude)
                            <= threshold) and (clusters_indexes[neighbor.Index] == NOT_IN_A_CLUSTER):
                        clusters_indexes[neighbor.Index] = current_cluster_index
            # Iterates the cluster index to the next cluster id.
            current_cluster_index += 1
    df.sort_index(inplace=True)
    df['cluster_id'] = clusters_indexes
    return df


# Clustering the data by k-means algorithms using latitude and longitude as features.
def clustering_by_kmeans(df, n_clusters=10):
    mat = df[['latitude', 'longitude']].values
    km = KMeans(n_clusters=n_clusters).fit(mat)
    df['cluster_id'] = km.labels_
    return df


# Clustering the data by the dbscan algorithm .
def clustering_by_dbscan(df, eps=0.001, min_samples=3):
    mat = df[['latitude', 'longitude']].values
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(mat)
    df['cluster_id'] = db.labels_
    return df


# Clustering the data by the aglomerative clustering (a hierarchical clustering) algorithm .
def clustering_by_agglomerative_clustering(df, n_clusters=2):
    mat = df[['latitude', 'longitude']].values
    ag = AgglomerativeClustering(n_clusters=n_clusters).fit(mat)
    df['cluster_id'] = ag.labels_
    return df
