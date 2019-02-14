import igraph
from utils.utils import distance_in_kilometers


# Creates a network using igraph library.
def create_network(n_vertices, id_attribute=None, attributes=None):
    g = igraph.Graph()
    g.add_vertices(n_vertices)
    # Assigns the vertices id attributes to the vertices of the network.
    if attributes:
        g.vs[id_attribute] = attributes
    return g


# Adds the edge list to the network.
def add_edges(g, edge_list):
    g.add_edges(edge_list)
    return g


# Adds weighted edges to the edge list of the network.
def add_weighted_edges(g, edge_list, edges_weights):
    g.add_edges(edge_list)
    g.es['weight'] = edges_weights
    return g


# Generates a local-local network from a pandas dataframe.
def convert_dataframe_to_local_local_network(df, id_attribute, latitude_attribute, longitude_attribute,
                                             threshold_distance_in_km):
    # Creates the network.
    attributes = list(df[id_attribute])
    local_local_network = create_network(len(df.index), id_attribute, attributes)
    latitudes = list(df[latitude_attribute])
    longitudes = list(df[longitude_attribute])
    edge_list = []
    edges_weights = []
    for v in range(local_local_network.vcount()):
        # node_position = (latitudes[v], longitudes[v])
        for u in range((v + 1), local_local_network.vcount()):
            distance = distance_in_kilometers(latitudes[v], longitudes[v], latitudes[u], longitudes[u])
            if distance <= threshold_distance_in_km:
                edge_list.append((v, u))
                edges_weights.append(distance)
    local_local_network.add_edges(edge_list)
    local_local_network.es['weight'] = edges_weights
    return local_local_network
