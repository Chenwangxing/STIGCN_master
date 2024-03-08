import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, in_dims=2, d_model=64, num_heads=4):
        super(SelfAttention, self).__init__()
        self.embedding = nn.Linear(in_dims, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.scaled_factor = torch.sqrt(torch.Tensor([d_model])).cuda()
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
    def split_heads(self, x):
        # x [batch_size seq_len d_model]
        x = x.reshape(x.shape[0], -1, self.num_heads, x.shape[-1] // self.num_heads).contiguous()
        return x.permute(0, 2, 1, 3)  # [batch_size nun_heads seq_len depth]
    def forward(self, x, mask=False, multi_head=False):
        # batch_size seq_len 2
        assert len(x.shape) == 3
        embeddings = self.embedding(x)  # batch_size seq_len d_model
        query = self.query(embeddings)  # batch_size seq_len d_model
        key = self.key(embeddings)      # batch_size seq_len d_model
        if multi_head:
            query = self.split_heads(query)  # B num_heads seq_len d_model
            key = self.split_heads(key)  # B num_heads seq_len d_model
            attention = torch.matmul(query, key.permute(0, 1, 3, 2))  # (batch_size, num_heads, seq_len, seq_len)
        else:
            attention = torch.matmul(query, key.permute(0, 2, 1))  # (batch_size, seq_len, seq_len)
        attention = self.softmax(attention / self.scaled_factor)
        if mask is True:
            mask = torch.ones_like(attention)
            attention = attention * torch.tril(mask)
        return attention


class ZeroSoftmax(nn.Module):
    def __init__(self):
        super(ZeroSoftmax, self).__init__()
    def forward(self, x, dim=0, eps=1e-5):
        x_exp = torch.pow(torch.exp(x) - 1, exponent=2)
        x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
        x = x_exp / (x_exp_sum + eps)
        return x


class SparseWeightedAdjacency(nn.Module):
    def __init__(self, spa_in_dims=2, tem_in_dims=3, embedding_dims=64, dropout=0,):
        super(SparseWeightedAdjacency, self).__init__()
        # dense interaction
        self.spatial_attention = SelfAttention(spa_in_dims, embedding_dims)
        self.temporal_attention = SelfAttention(tem_in_dims, embedding_dims)
        self.spatial_output = nn.Sigmoid()
        self.temporal_output = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # spatial fusion
        self.fc1s = nn.Conv2d(4, 16, 1, bias=False)
        self.relus = nn.ReLU()
        self.fc2s = nn.Conv2d(16, 4, 1, bias=False)
        self.ssigmoid = nn.Sigmoid()
        # temporal fusion
        self.fc1t = nn.Conv2d(4, 16, 1, bias=False)
        self.relut = nn.ReLU()
        self.fc2t = nn.Conv2d(16, 4, 1, bias=False)
        self.tsigmoid = nn.Sigmoid()

        self.dropout = dropout
        self.spa_softmax = nn.Softmax(dim=-1)
        self.tem_softmax = nn.Softmax(dim=-1)
        self.spatial_avgw = nn.Parameter(torch.ones(1), requires_grad=True)
        self.spatial_maxw = nn.Parameter(torch.ones(1), requires_grad=True)
        self.temporal_avgw = nn.Parameter(torch.ones(1), requires_grad=True)
        self.temporal_maxw = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, graph, identity):
        assert len(graph.shape) == 3
        spatial_graph = graph[:, :, 1:]  # (T N 2)
        temporal_graph = graph.permute(1, 0, 2)  # (N T 3)
        # (T num_heads N N)   (T N d_model)
        dense_spatial_interaction = self.spatial_attention(spatial_graph, multi_head=True)
        # (N num_heads T T)   (N T d_model)
        dense_temporal_interaction = self.temporal_attention(temporal_graph, multi_head=True)
        # (T num_heads 1 1)
        spatial_avg_act = self.fc2s(self.relus(self.fc1s(self.avg_pool(dense_spatial_interaction))))
        spatial_max_act = self.fc2s(self.relus(self.fc1s(self.max_pool(dense_spatial_interaction))))
        spatial_act = self.ssigmoid(self.spatial_maxw * spatial_max_act + self.spatial_avgw * spatial_avg_act)
        # (N num_heads 1 1)
        temporal_avg_act = self.fc2t(self.relut(self.fc1t(self.avg_pool(dense_temporal_interaction))))
        temporal_max_act = self.fc2t(self.relut(self.fc1t(self.max_pool(dense_temporal_interaction))))
        temporal_act = self.tsigmoid(self.temporal_maxw * temporal_max_act + self.temporal_avgw * temporal_avg_act)
        # (T num_heads N N)
        spatial_fusion_interaction = dense_spatial_interaction * temporal_act.permute(2, 1, 0, 3)
        # (N num_heads T T)
        temporal_fusion_interaction = dense_temporal_interaction * spatial_act.permute(2, 1, 0, 3)
        dense_spatial_interaction = torch.cat((dense_spatial_interaction, spatial_fusion_interaction), dim=1)
        dense_temporal_interaction = torch.cat((dense_temporal_interaction, temporal_fusion_interaction), dim=1)
        spatial_interaction_mask = self.spatial_output(dense_spatial_interaction)     # spatial_interaction_mask[T, 4, N, N]
        temporal_interaction_mask = self.temporal_output(dense_temporal_interaction)  # temporal_interaction_mask[N, 4, T, T]

        # self-connected
        spatial_interaction_mask = spatial_interaction_mask + identity[0].unsqueeze(1)
        temporal_interaction_mask = temporal_interaction_mask + identity[1].unsqueeze(1)
        normalized_spatial_adjacency_matrix = self.spa_softmax(spatial_interaction_mask)
        normalized_temporal_adjacency_matrix = self.tem_softmax(temporal_interaction_mask)

        return normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix


class GraphConvolution(nn.Module):
    def __init__(self, in_dims=2, embedding_dims=16, dropout=0):
        super(GraphConvolution, self).__init__()
        self.embedding = nn.Linear(in_dims, embedding_dims, bias=False)
        self.activation = nn.PReLU()
        self.dropout = dropout
    def forward(self, graph, adjacency):
        # graph [batch_size 1 seq_len 2]
        # adjacency [batch_size num_heads seq_len seq_len]
        gcn_features = self.embedding(torch.matmul(adjacency, graph))
        gcn_features = F.dropout(self.activation(gcn_features), p=self.dropout)
        return gcn_features  # [batch_size num_heads seq_len hidden_size]


class SparseGraphConvolution(nn.Module):
    def __init__(self, in_dims=16, embedding_dims=16, dropout=0):
        super(SparseGraphConvolution, self).__init__()
        self.dropout = dropout
        self.spatial_temporal_sparse_gcn = nn.ModuleList()
        self.temporal_spatial_sparse_gcn = nn.ModuleList()
        self.spatial_temporal_sparse_gcn.append(GraphConvolution(in_dims, embedding_dims))
        self.temporal_spatial_sparse_gcn.append(GraphConvolution(in_dims, embedding_dims))
    def forward(self, graph, normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix):
        # graph [1 seq_len num_pedestrians  3]
        # _matrix [batch num_heads seq_len seq_len]
        graph = graph[:, :, :, 1:]
        spa_graph = graph.permute(1, 0, 2, 3)  # (seq_len 1 num_p 2)
        tem_graph = spa_graph.permute(2, 1, 0, 3)  # (num_p 1 seq_len 2)
        gcn_spatial_features = self.spatial_temporal_sparse_gcn[0](spa_graph, normalized_spatial_adjacency_matrix)
        gcn_spatial_features = gcn_spatial_features.permute(2, 1, 0, 3)
        gcn_temporal_features = self.temporal_spatial_sparse_gcn[0](tem_graph, normalized_temporal_adjacency_matrix)
        return gcn_spatial_features, gcn_temporal_features


class TrajectoryModel(nn.Module):

    def __init__(self,
                 embedding_dims=64, number_gcn_layers=1, dropout=0,
                 obs_len=8, pred_len=12, n_tcn=5,
                 out_dims=5, num_heads=4):
        super(TrajectoryModel, self).__init__()
        self.number_gcn_layers = number_gcn_layers
        self.n_tcn = n_tcn
        self.dropout = dropout
        # graph learning
        self.sparse_weighted_adjacency_matrices = SparseWeightedAdjacency()
        # STIGCN (Part of the code comes from SGCN)
        self.stsgcn = SparseGraphConvolution(
            in_dims=2, embedding_dims=embedding_dims // num_heads, dropout=dropout
        )
        self.fusion_s = nn.Conv2d(num_heads * 2, num_heads, kernel_size=1, bias=False)
        self.fusion_t = nn.Conv2d(num_heads * 2, num_heads, kernel_size=1, bias=False)
        self.tcns = nn.ModuleList()
        self.tcns.append(nn.Sequential(
            nn.Conv2d(obs_len, pred_len, 3, padding=1),
            nn.PReLU()
        ))
        for j in range(1, self.n_tcn - 1):
            self.tcns.append(nn.Sequential(
                nn.Conv2d(pred_len, pred_len, 3, padding=1),
                nn.PReLU()
        ))
        self.TEP_tcn = nn.Sequential(
            nn.Conv2d(pred_len * 4, pred_len, 3, padding=1),
            nn.PReLU()
        )
        self.output = nn.Linear(embedding_dims // num_heads, out_dims)

    def forward(self, graph, identity):
        # graph 1 obs_len N 3
        normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix = \
            self.sparse_weighted_adjacency_matrices(graph.squeeze(), identity)
        gcn_spatial_features, gcn_temporal_features = self.stsgcn(
            graph, normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix
        )
        # gcn_representation [Num, heads, T, 16]
        gcn_representation = self.fusion_s(gcn_spatial_features) + self.fusion_t(gcn_temporal_features)
        gcn_representation = gcn_representation.permute(0, 2, 1, 3)
        features1 = self.tcns[0](gcn_representation)
        features2 = F.dropout(self.tcns[1](features1), p=self.dropout)
        features3 = F.dropout(self.tcns[2](features2), p=self.dropout)
        features4 = F.dropout(self.tcns[3](features3), p=self.dropout)
        features = torch.cat((features1, features2, features3, features4), dim=1)
        features = F.dropout(self.TEP_tcn(features), p=self.dropout)

        prediction = torch.mean(self.output(features), dim=-2)

        return prediction.permute(1, 0, 2).contiguous()
