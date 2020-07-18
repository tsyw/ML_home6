import numpy as np
from sklearn import preprocessing
import os


def load_w2v_feature(file, max_idx=0):
    with open(file, "rb") as f:
        nu = 0
        for line in f:
            content = line.strip().split()
            nu += 1
            if nu == 1:
                n, d = int(content[0]), int(content[1])
                feature = [[0.] * d for i in range(max(n, max_idx + 1))]
                continue
            index = int(content[0])
            while len(feature) <= index:
                feature.append([0.] * d)
            for i, x in enumerate(content[1:]):
                feature[index][i] = float(x)
    for item in feature:
        assert len(item) == d
    return np.array(feature, dtype=np.float32)


def load_data(embedding_dim=64, train_ratio= 75, valid_ratio=12.5):
    filedir = 'weibo'
    graphs = np.load(os.path.join(filedir, 'adjacency_matrix.npy')).astype(np.float32)
    # self-loop trick, the input graphs should have no self-loop
    graphs[graphs != 0] = 1.0

    """
    identity = np.identity(graphs.shape[1])
    graphs += identity
    if model == 'gat':
        graphs = graphs.astype(np.dtype('B'))
    elif model == 'gcn':
        # normalized graph laplacian for GCN: D^{-1/2}AD^{-1/2}
        for i in range(len(graphs)):
            graph = graphs[i]
            d_root_inv = 1. / np.sqrt(np.sum(graph, axis=1))
            graph = (graph.T * d_root_inv).T * d_root_inv
            graphs[i] = graph
    """
    influence_features = np.load(os.path.join(filedir, "influence_feature.npy")).astype(np.float32)
    labels = np.load(os.path.join(filedir, "label.npy"))
    vertices = np.load(os.path.join(filedir, "vertex_id.npy"))
    vertex_features = np.load(os.path.join(filedir, "vertex_feature.npy"))
    vertex_features = preprocessing.scale(vertex_features)
    embedding_path = os.path.join(filedir, "deepwalk.emb_%d" % embedding_dim)
    max_vertex_idx = np.max(vertices)
    embedding = load_w2v_feature(embedding_path, max_vertex_idx)
    N = graphs.shape[0]

    valid_start, test_start = int(N * train_ratio / 100), int(N * (train_ratio + valid_ratio) / 100)
    
    train_data = graphs[0:valid_start], influence_features[0:valid_start], labels[0:valid_start], vertices[0:valid_start]
    valid_data = graphs[valid_start:test_start], influence_features[valid_start:test_start], labels[valid_start:test_start], vertices[valid_start:test_start]
    test_data = graphs[test_start:], influence_features[test_start:], labels[test_start:],vertices[test_start:]

    return embedding, vertex_features, train_data, valid_data, test_data

