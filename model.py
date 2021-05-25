import torch
import math
import numpy as np
from operator import itemgetter
from weighted_gat import WeightedGATConv
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import GCNConv, GATConv, GatedGraphConv, SAGEConv
import torch.nn.functional as F
import torch.nn as nn


class Embedding2Score(nn.Module):
    def __init__(self, hidden_size):
        super(Embedding2Score, self).__init__()
        self.hidden_size = hidden_size
        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 1)
        self.W_3 = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def forward(self, session_embedding, all_item_embedding, batch):
        sections = torch.bincount(batch)
        v_i = torch.split(session_embedding, tuple(sections.cpu().numpy()))
        v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in v_i)

        alpha = self.q(torch.sigmoid(self.W_1(torch.cat(v_n_repeat, dim=0)) + self.W_2(session_embedding)))
        s_g_whole = alpha * session_embedding 
        s_g_split = torch.split(s_g_whole, tuple(sections.cpu().numpy())) 
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)

        v_n = tuple(nodes[-1].view(1, -1) for nodes in v_i)
        s_h = self.W_3(torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1))

        z_i_hat = torch.mm(s_h, all_item_embedding.weight.transpose(1, 0))
        
        return s_h, z_i_hat

class CGSR(torch.nn.Module):
    def __init__(self, args,device):
        super(CGSR, self).__init__()
        self.args=args
        self.device=device
        self.heads = args.WGAT_heads
        self.hidden_size = args.hidden_size
        self.causal_embedding = torch.nn.Embedding(args.item_num, embedding_dim=self.hidden_size)
        self.effect_embedding = torch.nn.Embedding(args.item_num, embedding_dim=self.hidden_size)
        self.relation_embedding = torch.nn.Embedding(args.item_num, embedding_dim=self.hidden_size)
        self.relation_data = torch.load(f'./datasets/{args.dataset}/'+'relation_graph.pt')
        self.WGAT1 = WeightedGATConv(in_channels=args.hidden_size,
                                     out_channels=args.hidden_size,
                                     heads=self.heads,
                                     concat=False,
                                     negative_slope=args.leaky_relu,
                                     dropout=args.dropout,
                                     bias=True,
                                     weighted=True,
                                     device=self.device)
        self.WGAT2 = WeightedGATConv(in_channels=args.hidden_size,
                                     out_channels=args.hidden_size,
                                     heads=self.heads,
                                     concat=False,
                                     negative_slope=args.leaky_relu,
                                     dropout=args.dropout,
                                     bias=True,
                                     weighted=True,
                                     device=self.device)
        self.WGAT3 = WeightedGATConv(in_channels=args.hidden_size,
                                     out_channels=args.hidden_size,
                                     heads=self.heads,
                                     concat=False,
                                     negative_slope=args.leaky_relu,
                                     dropout=args.dropout,
                                     bias=True,
                                     weighted=True,
                                     device=self.device)
        self.item_linear = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.fuse_weight1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.edge_weight1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.edge_weight2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.edge_weight3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.e2s1 = Embedding2Score(self.hidden_size)
        self.e2s2 = Embedding2Score(self.hidden_size)
        self.e2s3 = Embedding2Score(self.hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        self.fuse_weight1.data.fill_(1.)
        self.fuse_weight2.data.fill_(1.)
        self.fuse_weight3.data.fill_(1.)
        self.edge_weight1.data.fill_(0.5)
        self.edge_weight2.data.fill_(-0.5)
        self.edge_weight3.data.fill_(-0.5)
        self.effect_embedding.weight=self.causal_embedding.weight
        self.relation_embedding.weight=self.causal_embedding.weight

    def forward(self, data):
        inputs, edge_index, batch, edge_attr_eff, edge_attr_cau= data.x, data.edge_index, data.batch, data.edge_attr, data.edge_attr_eff
        edge_index_cau=edge_index[[1,0],:]

        x = self.causal_embedding(inputs-1).squeeze(axis=1)
        x = self.WGAT1(x, edge_index_cau, edge_attr_cau)

        e_x = self.effect_embedding(inputs-1).squeeze(axis=1)
        e_x = self.WGAT2(e_x, edge_index, edge_attr_eff)

        cause_star, cause_scores = self.e2s1(x, self.effect_embedding, batch)
        effect_star, effect_scores = self.e2s2(e_x, self.causal_embedding, batch)

        relation_data = self.relation_data
        relation_x = relation_data.x.to(self.device)
        relation_edge_index = relation_data.edge_index
        relation_edge_attr = relation_data.edge_attr.to(self.device)
        relation_edge_tri_attr1 = relation_data.edge_tri_attr1.to(self.device)
        relation_edge_tri_attr2 = relation_data.edge_tri_attr2.to(self.device)
        relation_edge_tri_attr3 = relation_data.edge_tri_attr3.to(self.device)
        relation_nodes=relation_data.node_dict
        batch_nodes = np.unique(inputs.cpu().detach().numpy().flatten())
        node_idx=torch.tensor(itemgetter(*batch_nodes)(relation_nodes), dtype=torch.long)

        subgraph_loaders = NeighborSampler(relation_edge_index, node_idx=node_idx, sizes=[-1], shuffle=False, num_workers=0, batch_size=batch_nodes.shape[0])
        for b_size, n_id, adjs in subgraph_loaders:
            adjs=adjs.to(self.device)
            n_id = n_id.to(self.device)
            edge_index, e_id =adjs.edge_index,adjs.e_id
            relation_edge_attr=relation_edge_attr[e_id]
            relation_edge_tri_attr1=relation_edge_tri_attr1[e_id]
            relation_edge_tri_attr2=relation_edge_tri_attr2[e_id]
            relation_edge_tri_attr3=relation_edge_tri_attr3[e_id]
            edge_index_np=torch.index_select(n_id, 0, edge_index.flatten()).cpu().detach().numpy().reshape(2,-1)
            indices = np.setdiff1d(np.arange(edge_index_np.shape[1]),np.where(1-np.isin(edge_index_np,node_idx.numpy()))[1])
            edge_index=edge_index[:,indices]
            edge_index=torch.cat([edge_index,edge_index[[1,0],:]],dim=1)
            relation_edge_attr=torch.cat([relation_edge_attr[indices],relation_edge_attr[indices]],dim=0)
            relation_edge_tri_attr1=torch.cat([relation_edge_tri_attr1[indices],relation_edge_tri_attr1[indices]],dim=0)
            relation_edge_tri_attr2=torch.cat([relation_edge_tri_attr2[indices],relation_edge_tri_attr2[indices]],dim=0)
            relation_edge_tri_attr3=torch.cat([relation_edge_tri_attr3[indices],relation_edge_tri_attr3[indices]],dim=0)
            r_edge_attr=relation_edge_attr+self.edge_weight1*relation_edge_tri_attr1-self.edge_weight2*relation_edge_tri_attr2-self.edge_weight3*relation_edge_tri_attr3

            r_x = self.relation_embedding(relation_x[n_id]-1).squeeze(axis=1)
            r_x = self.WGAT3(r_x, edge_index, r_edge_attr)

        indices = []
        relation_x=relation_x[n_id].squeeze()
        for i in inputs:
            indices.append((relation_x==i).nonzero(as_tuple=False).flatten()[0])
        indices = torch.tensor(indices).to(self.device)
        r_x = torch.index_select(r_x, 0, indices)

        relation_star, relation_scores = self.e2s3(r_x, self.relation_embedding, batch)

        item_star=self.item_linear((cause_star+effect_star+relation_star)/3)
        item_embedding=(self.causal_embedding.weight+self.effect_embedding.weight+self.relation_embedding.weight)/3
        item_scores = item_star @ item_embedding.T

        scores=item_scores+self.fuse_weight2*(cause_scores-self.fuse_weight1*effect_scores)+self.fuse_weight3*relation_scores
        return scores
