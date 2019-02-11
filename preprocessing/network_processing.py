import igraph
from utils.utils import distance_in_kilometers


# Generates a local-local network from a pandas dataframe.
def dataframe_2_network(df, id_attribute, latitude_attribute, longitude_attribute,
                        threshold_distance_in_km):
    local_local_network = create_nodes(df, id_attribute)
    local_local_network = threshold_distance_edges(df, local_local_network, latitude_attribute,
                                                   longitude_attribute, threshold_distance_in_km)
    return local_local_network


# Creates the network, adds the nodes and assigns the list of attributes according to the argument.
def create_nodes(df, id_attribute):
    # Creates the network.
    g = igraph.Graph()
    # Adds the vertices to the network.
    g.add_vertices(len(df.index))
    # Assigns the vertices id attributes to the vertices of the network.
    g.vs[id_attribute] = list(df[id_attribute])
    return g


# Creates the network, adds the nodes and assigns the list of attributes according to the argument.
def threshold_distance_edges(df, graph, latitude_attribute, longitude_attribute,
                             threshold_distance_in_km):
    latitudes = list(df[latitude_attribute])
    longitudes = list(df[longitude_attribute])
    edge_list = []
    edges_weights = []
    for v in range(graph.vcount()):
        # node_position = (latitudes[v], longitudes[v])
        for u in range((v + 1), graph.vcount()):
            distance = distance_in_kilometers(latitudes[v], longitudes[v], latitudes[u], longitudes[u])
            if distance <= threshold_distance_in_km:
                edge_list.append((v, u))
                edges_weights.append(distance)
    graph.add_edges(edge_list)
    graph.es['weight'] = edges_weights
    return graph
