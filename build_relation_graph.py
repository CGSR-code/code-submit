import torch
import numpy as np
import networkx as nx
import pickle
import argparse
import os
from tqdm import tqdm
from collections import defaultdict
from torch_geometric.data import Data


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/gowalla/amazon')
args = parser.parse_args()
dataset = './datasets/'+args.dataset

# relation graph
with open(dataset+'all_train_seq.txt', 'rb') as f:
    train = pickle.load(f)
if not os.path.exists(dataset+'unique_nodes.pkl'):
    # unique items in train
    unique_nodes = list(set([x for y in train for x in y]))
    pickle.dump(unique_nodes, open(dataset+'unique_nodes.pkl', 'wb'))
else:
    unique_nodes = pickle.load(open(dataset+'unique_nodes.pkl', 'rb'))
print("len unique_nodes:",len(unique_nodes))

nodes = {unique_nodes[i]: i for i in range(len(unique_nodes))}

G1 = nx.Graph()
G2 = nx.DiGraph()

out_degree_inv = defaultdict(int)
in_degree_inv = defaultdict(int)
pair = defaultdict(int)
for sequence in train:
    senders = []
    for node in sequence:
        senders.append(node)
    receivers = senders[:]
    if len(senders) != 1:
        del senders[-1]
        del receivers[0] 
        for sender, receiver in zip(senders, receivers):
            out_degree_inv[sender]+=1
            in_degree_inv[receiver]+=1
            pair[str(sender) + '-' + str(receiver)] += 1
            G1.add_edge(sender, receiver)
            G2.add_edge(sender, receiver)

epsilon = 2
graph_node=defaultdict(list)
for i, seq in enumerate(train):
    if len(seq) > 0:
        for j, node in enumerate(seq):
            if j>len(seq)-epsilon-1:
                graph_node[node]+=seq[j+1:len(seq)]
            else:
                graph_node[node]+=seq[j+1:j+3]

edge_lists = []
l=[]
for i, k in tqdm(enumerate(graph_node.keys())):
    if len(graph_node[k]) > 0:
        edge_lists+=[[min(k,j),max(k,j)] for j in graph_node[k]]
edge_lists=[list(x) for x in set(tuple(x) for x in edge_lists)]


def findPaths(G,u,n):
    if n==0:
        return [[u]]
    paths = [[u]+path for neighbor in G.neighbors(u) for path in findPaths(G,neighbor,n-1) if u not in path]
    return paths

all_tri_paths = []
for node in G1:
    all_tri_paths.extend(findPaths(G1,node,2))

all_tri_paths=set([(min(path[0], path[2]),path[1],max(path[0], path[2])) for path in all_tri_paths])
all_tri_paths=[list(path) for path in all_tri_paths]

dou_weight = defaultdict(float)
tri_weight1 = defaultdict(float)
tri_weight2 = defaultdict(float)
tri_weight3 = defaultdict(float)

for key in pair.keys():
    dou_weight[key]+=2*pair[key]/(out_degree_inv[int(key.split('-')[0])]+in_degree_inv[int(key.split('-')[1])])
    dou_weight[key.split('-')[1]+'-'+key.split('-')[0]]+=dou_weight[key]

for i,path in enumerate(all_tri_paths):
    if G2.has_edge(path[0], path[1]) and G2.has_edge(path[1],path[2]):
        tri_weight1[str(path[0])+'-'+str(path[2])]+=(pair[str(path[0])+'-'+str(path[1])]+pair[str(path[1])+'-'+str(path[2])])/(out_degree_inv[path[0]]+in_degree_inv[path[2]])
        tri_weight1[str(path[2])+'-'+str(path[0])]+=(pair[str(path[0])+'-'+str(path[1])]+pair[str(path[1])+'-'+str(path[2])])/(out_degree_inv[path[0]]+in_degree_inv[path[2]])
    if G2.has_edge(path[2], path[1]) and G2.has_edge(path[1],path[0]):
        tri_weight1[str(path[2])+'-'+str(path[0])]+=(pair[str(path[2])+'-'+str(path[1])]+pair[str(path[1])+'-'+str(path[0])])/(out_degree_inv[path[2]]+in_degree_inv[path[0]])
        tri_weight1[str(path[0])+'-'+str(path[2])]+=(pair[str(path[2])+'-'+str(path[1])]+pair[str(path[1])+'-'+str(path[0])])/(out_degree_inv[path[2]]+in_degree_inv[path[0]])
    if G2.has_edge(path[1], path[0]) and G2.has_edge(path[1],path[2]):
        tri_weight2[str(path[0])+'-'+str(path[2])]+=(pair[str(path[1])+'-'+str(path[0])]+pair[str(path[1])+'-'+str(path[2])])/(in_degree_inv[path[0]]+in_degree_inv[path[2]])
        tri_weight2[str(path[2])+'-'+str(path[0])]+=(pair[str(path[1])+'-'+str(path[0])]+pair[str(path[1])+'-'+str(path[2])])/(in_degree_inv[path[0]]+in_degree_inv[path[2]])
    if G2.has_edge(path[0], path[1]) and G2.has_edge(path[2],path[1]):
        tri_weight3[str(path[0])+'-'+str(path[2])]+=(pair[str(path[0])+'-'+str(path[1])]+pair[str(path[2])+'-'+str(path[1])])/(out_degree_inv[path[0]]+out_degree_inv[path[2]])
        tri_weight3[str(path[2])+'-'+str(path[0])]+=(pair[str(path[0])+'-'+str(path[1])]+pair[str(path[2])+'-'+str(path[1])])/(out_degree_inv[path[0]]+out_degree_inv[path[2]])



l = np.array(edge_lists)  # sort by first then second column
r = np.array([l[:, 1], l[:, 0]]).transpose()
l=np.vstack((l,r))
l=np.array([list(x) for x in set(tuple(x) for x in l)])

edge_attr,edge_tri_attr1,edge_tri_attr2,edge_tri_attr3=[],[],[],[]
for edge in l:
    edge_attr.append(dou_weight[str(edge[0])+'-'+str(edge[1])])
    edge_tri_attr1.append(tri_weight1[str(edge[0])+'-'+str(edge[1])])
    edge_tri_attr2.append(tri_weight2[str(edge[0])+'-'+str(edge[1])])
    edge_tri_attr3.append(tri_weight3[str(edge[0])+'-'+str(edge[1])])

print("edge_attr.count(0):",edge_attr.count(0.),edge_attr[:10])
edge_attr = torch.tensor(edge_attr,dtype=torch.float)
edge_tri_attr1 = torch.tensor(edge_tri_attr1,dtype=torch.float)
edge_tri_attr2 = torch.tensor(edge_tri_attr2,dtype=torch.float)
edge_tri_attr3 = torch.tensor(edge_tri_attr3,dtype=torch.float)

l=np.array([np.array(list(map(lambda x:nodes[x],edge))) for edge in l])
edge_index = torch.from_numpy(l).long()

x=torch.tensor([[i] for i in unique_nodes],dtype=torch.long)
data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr,edge_tri_attr1=edge_tri_attr1,edge_tri_attr2=edge_tri_attr2,edge_tri_attr3=edge_tri_attr3, node_dict=nodes)

torch.save(data, dataset + 'relation_graph.pt')
