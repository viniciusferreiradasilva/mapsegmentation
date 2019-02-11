#!/usr/bin/python
import matplotlib.pyplot as plt
import folium

from folium.plugins import FastMarkerCluster


# Draw a simple map with matplotlib.
def draw_cluster_map(data):
    ax = plt.scatter(data.latitude, data.longitude, c=data.cluster_id)
    plt.show()


# Draw an fast enriched clusterized map using the folium library withouth pop-ups.
def draw_fast_cluster_map(data):
    m = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=4)
    m.add_child(FastMarkerCluster(data=data[['latitude', 'longitude']].values.tolist()))
    m.save(outfile='map.html')
