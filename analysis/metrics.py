#!/usr/bin/python
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

"""
File containing metrics to analyse clustering algorithms outputs.
"""


def silhouette_coefficient(df):
    """Calculates the clustering silhouette coefficient for the clusterized data.

    Parameters
    ----------

    df : A pandas dataframe containing the latitude, longitude and cluster id data that will be clusterized.
    The columns of the dataframe must have the names 'latitude', 'longitude' and 'cluster_id'.

    Returns
    -------
    float representing the value of silhouette coefficient for that clustering.
    """
    return silhouette_score(df[['latitude', 'longitude']].values, df['cluster_id'])


def silhouette_sample(df):
    """Calculates the clustering silhouette coefficient for each sample of the clusterized data.

    Parameters
    ----------

    df : A pandas dataframe containing the latitude, longitude and cluster id data that will be clusterized.
    The columns of the dataframe must have the names 'latitude', 'longitude' and 'cluster_id'.

    Returns
    -------
    list of float representing the value of silhouette coefficient for sample in it's cluster.
    """
    return silhouette_samples(df[['latitude', 'longitude']].values, df['cluster_id'])
