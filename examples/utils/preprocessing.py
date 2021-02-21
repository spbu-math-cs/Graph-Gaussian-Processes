import pandas as pd
import networkx as nx
import numpy as np
import os
import pickle
import zipfile
import osmnx
from shapely.geometry import Point
from shapely.ops import nearest_points
from .utils import cut


def load_cora(num_train, num_test):
    #unzip data
    with zipfile.ZipFile('data/cora.zip', 'r') as zip_ref:
        zip_ref.extractall('data')
    # data reading
    data_dir = 'data/cora'
    edgelist = pd.read_csv(os.path.join(data_dir, "cora.cites"), sep='\t', header=None, names=["target", "source"])
    edgelist["label"] = "cites"
    # Graph creation
    G = nx.from_pandas_edgelist(edgelist, edge_attr="label")
    nx.set_node_attributes(G, "paper", "label")
    G = nx.Graph(G.subgraph(max(nx.connected_components(G), key=len)))
    # Labels extraction
    node_data = pd.read_csv(os.path.join(data_dir, "cora.content"), sep='\t', header=None, index_col=0)
    # Our GP model doesn't use bag of words
    node_subject = pd.DataFrame(node_data.iloc[:, -1]).rename({1434: ''}, axis='columns')

    # Nodes relabeling with numbers from 0 to len(G)-1
    names_to_id = {}
    nodelist = []
    for vertex in G.nodes:
        nodelist.append(len(names_to_id))
        names_to_id[vertex] = len(names_to_id)
    nx.relabel_nodes(G, names_to_id, copy=False)
    G = nx.from_numpy_matrix(nx.to_numpy_array(G, nodelist=nodelist))

    # Replacing classes with their identifiers
    classes = pd.get_dummies(node_subject, prefix_sep='')
    cls_to_id = {}
    for cls in classes.columns:
        cls_to_id[cls] = len(cls_to_id)

    # Dataset creation
    y = np.zeros((len(G), 1))
    x = np.arange(len(G), dtype=np.float64)
    x = x.reshape(-1, 1)

    for row_name, row in classes.iterrows():
        if names_to_id.get(row_name) is not None:
            numpy_row = np.array(row.to_numpy(), dtype=np.int32)
            for i in range(7):
                if numpy_row[i] == 1:
                    y[names_to_id[row_name]] = i

    # Adding ground truth label for visualisation in yEd
    ground_truth = {int(x[i][0]): y[i][0] for i in range(len(x))}
    nx.set_node_attributes(G, values=ground_truth, name='ground_truth')

    # Splitting data into train and test
    random_permutation = np.random.permutation(len(G))
    train_id, test_id = random_permutation[:num_train], random_permutation[num_train:num_train+num_test]
    x_train, x_test = x[train_id], x[test_id]
    y_train, y_test = y[train_id], y[test_id]

    return G, (x_train, y_train), (x_test, y_test)


def load_PEMS(num_train=250, dtype=np.float64):
    #unzip daata
    with zipfile.ZipFile('data/PEMS.zip', 'r') as zip_ref:
        zip_ref.extractall('data')
    # Data reading
    with open('data/PEMS/adj_mx_bay.pkl', 'rb') as f:
        sensor_ids, sensor_id_to_ind, _ = pickle.load(f, encoding='latin1')
    all_signals = pd.read_hdf('data/PEMS/pems-bay.h5')
    coords = pd.read_csv('data/PEMS/graph_sensor_locations_bay.csv', header=None)

    # Loading real world graph of roads
    north, south, east, west = 37.450, 37.210, -121.80, -122.10
    if not os.path.isfile('data/PEMS/bay_graph.pkl'):
        cf = '["highway"~"motorway|motorway_link"]'  # Road filter, we don't use small ones.
        G = osmnx.graph_from_bbox(north=north, south=south, east=east, west=west, simplify=True, custom_filter=cf)
        with open('data/PEMS/bay_graph.pkl', 'wb') as f:  # frequent loading of maps leads to a temporal ban
            pickle.dump(G, f)
    else:
        with open('data/PEMS/bay_graph.pkl', 'rb') as f:
            G = pickle.load(f)

    G = osmnx.get_undirected(G)  # Matern GP supports only undirected graphs.

    # Graph cleaning up
    for _ in range(2):
        out_degree = G.degree
        to_remove = [node for node in G.nodes if out_degree[node] == 1]
        G.remove_nodes_from(to_remove)
    G = nx.convert_node_labels_to_integers(G)
    G.remove_nodes_from([372, 286])
    G = nx.convert_node_labels_to_integers(G)

    num_points = len(sensor_ids)

    np_coords = np.zeros((num_points, 2))  # Vector of sensors coordinates.
    for i in range(num_points):
        sensor_id, x, y = coords.iloc[i]
        ind = sensor_id_to_ind[str(int(sensor_id))]
        np_coords[ind][0], np_coords[ind][1] = x, y
    coords = np_coords

    sensor_ind_to_node = {}
    # Inserting sensors into a graph. During insertion, the edge containing the sensor is cut
    for point_id in range(num_points):
        sensor_ind_to_node[point_id] = len(G)  # adding new vertex at the end
        sensor_point = Point(coords[point_id, 1], coords[point_id, 0])
        u, v, key, geom = osmnx.get_nearest_edge(G, (sensor_point.y, sensor_point.x), return_geom=True)
        edge = G.edges[(u, v, key)]
        G.remove_edge(u, v, key)
        edge_1_geom, edge_2_geom = cut(geom, geom.project(sensor_point))
        l_ratio = geom.project(sensor_point, normalized=True)
        l_1, l_2 = l_ratio*edge['length'], (1-l_ratio)*edge['length']
        new_vertex = nearest_points(geom, sensor_point)[0]
        G.add_node(len(G), x=new_vertex.x, y=new_vertex.y)
        G.add_edge(u, len(G)-1, length=l_1, geometry=edge_1_geom)
        G.add_edge(len(G)-1, v, length=l_2, geometry=edge_2_geom)

    # Weights are inversely proportional to the length of the road
    lengths = nx.get_edge_attributes(G, 'length')
    lengths_list = [length for length in lengths.values()]
    mean_length = np.mean(lengths_list)
    weights = {}
    for edge, length in lengths.items():
        weights[edge] = mean_length / length
    nx.set_edge_attributes(G, values=weights, name='weight')

    # Sorry for that,
    # sensor_ind - sensor id in California database, sensor_id - local numeration (from 1 to num of sensors)
    sensor_id_to_node = {}
    for sensor_ind, node in sensor_ind_to_node.items():
        sensor_id = sensor_ids[sensor_ind]
        sensor_id_to_node[sensor_id] = node

    # Selecting signals at some moment
    signals = all_signals[
        (all_signals.index.weekday == 0) & (all_signals.index.hour == 17) & (all_signals.index.minute == 30)]

    # Dataset creation
    x, y = [], []
    for i in range(len(signals)):
        for sensor_id in sensor_ids:
            if sensor_id_to_node.get(sensor_id) is not None:
                node = sensor_id_to_node[sensor_id]
                signal = signals.iloc[i][int(sensor_id)]
                x.append([i, node])
                y.append([signal])

    x, y = np.asarray(x, dtype=dtype), np.asarray(y, dtype=dtype)
    x, y = x[:num_points, 1:], y[:num_points]

    # Splitting data into train and test
    random_perm = np.random.permutation(np.arange(x.shape[0]))
    train_vertex, test_vertex = random_perm[:num_train], random_perm[num_train:]
    x_train, x_test = x[train_vertex], x[test_vertex]
    y_train, y_test = y[train_vertex], y[test_vertex]

    return G, (x_train, y_train), (x_test, y_test), (x, y)
