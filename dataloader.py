import pickle
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data
from collections import defaultdict
from operator import itemgetter


class MultiSessionsGraph(InMemoryDataset):
    """Every session is a graph."""

    def __init__(self, root, phrase, transform=None, pre_transform=None):
        assert phrase in ['train', 'test']
        self.phrase = phrase
        super(MultiSessionsGraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.phrase + '.txt']

    @property
    def processed_file_names(self):
        return [self.phrase + '.pt']

    def download(self):
        pass

    def process(self):
        data = pickle.load(open(f"{self.root}/{self.phrase}.txt", "rb"))
        tra_seqs = pickle.load(open(f"{self.root}/all_train_seq.txt", "rb"))

        pair = defaultdict(int)
        tri_pair = defaultdict(int)
        out_degree_inv = defaultdict(int)
        in_degree_inv = defaultdict(int)
        node_to_id = {}
        edge_attr_cause={}
        edge_attr_effect={}
        i = 0
        G1 = nx.Graph()
        G2 = nx.DiGraph()
        for sequence in tra_seqs:
            senders = []
            for node in sequence:
                if node not in node_to_id:
                    node_to_id[node] = i
                    i += 1
                senders.append(node_to_id[node])
            for j in range(len(senders)-2):
                tri_pair[str(senders[j]) + '-' +str(senders[j+1]) + '-' + str(senders[j+2])]+=1
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
        
        allpaths = []
        for node in G1:
            allpaths.extend(findPaths(G1,node,2))
        backdoor_path=[]
        for i,path in enumerate(allpaths):
            if G2.has_edge(path[0], path[2]) and G2.has_edge(path[1],path[0]) and G2.has_edge(path[1],path[2]):
                backdoor_path.append(path)

        for j,path in enumerate(backdoor_path):
            pair[str(path[0]) + '-' + str(path[2])]=pair[str(path[0]) + '-' + str(path[2])] - tri_pair[str(path[1]) + '-' + str(path[0]) + '-' + str(path[2])]
    
        for key in pair.keys():
            edge_attr_cause[key] = pair[key]/out_degree_inv[int(key.split('-')[0])]
            edge_attr_effect[key] = pair[key]/in_degree_inv[int(key.split('-')[1])]
        
        # for key in pair.keys():
        #     edge_attr_cause[key] = 1.0
        #     edge_attr_effect[key] = 1.0
        
        data_list = []
        test_unexist=0
        for sequence, y in zip(data[0], data[1]):
            i = 0
            flag=0
            nodes = {}
            senders = []
            x = []
            for node in sequence:
                if node not in nodes:
                    nodes[node] = i
                    x.append([node])
                    i += 1
                senders.append(nodes[node])
            receivers = senders[:]
            if len(senders) != 1:
                del senders[-1]
                del receivers[0]
                res_nodes={value:key for key, value in nodes.items()}
                if self.phrase=='train':
                    edge_attr = torch.tensor([edge_attr_cause[str(node_to_id[res_nodes[senders[i]]]) + '-' + str(node_to_id[res_nodes[receivers[i]]])] for i in range(len(senders))],dtype=torch.float)
                    edge_attr_eff=torch.tensor([edge_attr_effect[str(node_to_id[res_nodes[senders[i]]]) + '-' + str(node_to_id[res_nodes[receivers[i]]])] for i in range(len(senders))],dtype=torch.float)
                else:
                    edge_attr=[]
                    edge_attr_eff=[]
                    for i in range(len(senders)):
                        if res_nodes[senders[i]] not in node_to_id.keys():
                            flag=1
                            test_unexist+=1
                            break
                        if res_nodes[receivers[i]] not in node_to_id.keys():
                            flag=1
                            test_unexist+=1
                            break
                        key=str(node_to_id[res_nodes[senders[i]]]) + '-' + str(node_to_id[res_nodes[receivers[i]]])
                        if key in edge_attr_cause.keys():
                            edge_attr.append(edge_attr_cause[key])
                            edge_attr_eff.append(edge_attr_effect[key])
                        else:
                            edge_attr.append(1/(1+out_degree_inv[node_to_id[res_nodes[senders[i]]]]))
                            edge_attr_eff.append(1/(1+in_degree_inv[node_to_id[res_nodes[receivers[i]]]]))
                    edge_attr = torch.tensor(edge_attr,dtype=torch.float)
                    edge_attr_eff=torch.tensor(edge_attr_eff,dtype=torch.float)
            # senders=itemgetter(*senders)(nodes)
            # receivers=itemgetter(*receivers)(nodes)
            else:
                edge_attr = torch.tensor([0.],dtype=torch.float)
                edge_attr_eff=torch.tensor([0.],dtype=torch.float)
            if flag==1:
                break
            edge_index = torch.tensor([senders, receivers], dtype=torch.long)
            x = torch.tensor(x, dtype=torch.long)
            y = torch.tensor([y], dtype=torch.long)
            #sequence = torch.tensor(sequence, dtype=torch.long)
            session_graph = Data(x=x, y=y,edge_index=edge_index, edge_attr=edge_attr,edge_attr_eff=edge_attr_eff)
            data_list.append(session_graph)

        print("len(test_unexist)",test_unexist)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def findPaths(G,u,n):
    if n==0:
        return [[u]]
    paths = [[u]+path for neighbor in G.neighbors(u) for path in findPaths(G,neighbor,n-1) if u not in path]
    return paths
