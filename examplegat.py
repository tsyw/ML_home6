import numpy as np
from utils import  load_data
from gat import BatchGAT
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

"""
 |V|: total number of nodes of the social network
 m: the number of instances, each instance is an ego-network of a user samlped from the social network.
 n: node number of the sub-network

 embeddings: pretrained node embeddings with shape (|V|, 64)
 vertex_features: vertext features with shape (|V|, 7)
 train_data: training dataset
 valid_data: validation dataset
 test_data: test dataset
 *_graphs: the sampled sub-graphs of a user, which is represented as adjacency matrix. shape: (m, n, n)
 *_inf_features: two dummy features indicating whether the user is active and whether the user is the ego. shape: (m,n,2)
 
 *_vertices: node ids of the sampled ego-network, each id is a value from 0 to |V|-1. shape:(m,n)
 *_labels: corresponding label of each instance. shape:(m,)
"""
cuda = True
#seed = np.random.seed(42)
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)
embeddings, vertex_features, train_data, valid_data, test_data = load_data(64)
epochs = 500
batch_size = 1024
heads = 8,8,1
embeddings = torch.FloatTensor(embeddings)
vertex_features = torch.FloatTensor(vertex_features)

train_graphs, train_inf_features, train_labels, train_vertices= train_data

valid_graphs, valid_inf_features, valid_labels, valid_vertices= valid_data
test_graphs, test_inf_features, test_labels, test_vertices= test_data
print(vertex_features.shape)
print(embeddings.shape)
print(train_graphs.shape)
print(train_vertices.shape)
print(train_inf_features.shape)
print(train_labels.shape)

def find_list(indices, data):
    out = []
    for i in indices:
        out = out + [data[i]]
    return np.array(out)

def batch_loader(data):
    rows = data[0].shape[0]
    indices = list(range(rows))
    while True:
        if batch_size < len(indices):
            batch_indices = indices[0:batch_size]  # 产生一个batch的index
        else:
            batch_indices = indices[0: len(indices)]
        if batch_size < len(indices):
            indices = indices[batch_size:]
        else:
            indices = []
        batch_d = []
        for d in data:
            temp_d = find_list(batch_indices, d)
            batch_d.append(temp_d)
        yield batch_d

"""
acquire the corresponding vertex features and embeddings of an instance.
"""
#vertex_features[train_vertices[0],:]
#embeddings[train_vertices[0],:]
n_classes = 2
class_weight = train_graphs.shape[0]/(n_classes * np.bincount(train_labels))
class_weight = torch.FloatTensor(class_weight)
hidden_units = 16,16
feature_dim = train_inf_features.shape[-1]
n_units = [feature_dim] + [x for x in hidden_units] + [n_classes]
n_heads = [x for x in heads]
model = BatchGAT(pretrained_emb=embeddings, vertex_feature=vertex_features, use_vertex_feature=True, n_units=n_units,n_heads=n_heads, dropout=0.2,instance_normalization=True)

model.cuda()
class_weight = class_weight.cuda()
params = [{'params': model.layer_stack.parameters()}]
optimizer = optim.Adagrad(params, lr=0.1, weight_decay=5e-4)

def evaluate(epoch, loader, thr=None, return_best_thr=False):
    model.eval()
    total = 0.
    loss, prec, rec, f1 = 0., 0., 0., 0.
    y_true, y_pred, y_score = [], [], []
    for i_batch, batch in enumerate(batch_loader(loader)):
        graph, features, labels, vertices = batch
        bs = graph.shape[0]
        features = torch.FloatTensor(features)
        graph = torch.FloatTensor(graph)
        labels = torch.LongTensor(labels)
        vertices = torch.LongTensor(vertices)
        features = features.cuda()
        graph = graph.cuda()
        labels = labels.cuda()
        vertices = vertices.cuda()

        output = model(features, vertices, graph)
        output = output[:, -1, :]
        loss_batch = F.nll_loss(output, labels, class_weight)
        loss += bs * loss_batch.item()

        y_true += labels.data.tolist()
        y_pred += output.max(1)[1].data.tolist()
        y_score += output[:, 1].data.tolist()
        total += bs
        if bs < batch_size:
            break
    model.train()

    if thr is not None:
        print("using threshold %.4f" %(thr))
        y_score = np.array(y_score)
        y_pred = np.zeros_like(y_score)
        y_pred[y_score > thr] = 1

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)
    print("loss: %.4f AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f" % (loss/total, auc, prec, rec, f1))
    if return_best_thr:
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        f1s = 2 * precs * recs / (precs + recs)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(f1s)]
        f1s = f1s[~np.isnan(f1s)]
        best_thr = thrs[np.argmax(f1s)]
        print("best threshold = %4f, f1 = %.4f"  % (best_thr, np.max(f1s)))
        return best_thr
    else:
        return None


def train(epoch, train_loader, valid_loader, test_loader):
    model.train()

    loss = 0.
    total = 0.
    for i_batch, batch in enumerate(batch_loader(train_loader)):
        graph, features, labels, vertices = batch
        bs = graph.shape[0]
        features = torch.FloatTensor(features)
        graph = torch.FloatTensor(graph)
        labels = torch.LongTensor(labels)
        vertices = torch.LongTensor(vertices)
        features = features.cuda()
        graph = graph.cuda()
        labels = labels.cuda()
        vertices = vertices.cuda()

        optimizer.zero_grad()
        output = model(features, vertices, graph)
        output = output[:, -1, :]
        loss_train = F.nll_loss(output, labels, class_weight)
        loss += bs * loss_train.item()
        total += bs
        loss_train.backward()
        optimizer.step()
        if bs < batch_size:
            break
    print("train loss in this epoch %.4f" % (loss/total))
    if (epoch + 1) % 10 == 0:
        print("epoch %d, checkpoint!" %(epoch))
        best_thr = evaluate(epoch, valid_loader, return_best_thr=True)
        evaluate(epoch, test_loader, thr=best_thr)


# Train model
t_total = time.time()
for epoch in range(epochs):
    train(epoch, train_data, valid_data, test_data)
# epoch 1: loss/total
print("optimization Finished")

best_thr = evaluate(epochs, valid_data, return_best_thr=True)

# Testing
evaluate(epochs, test_data, thr=best_thr)
