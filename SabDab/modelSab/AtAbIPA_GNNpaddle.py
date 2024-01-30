import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim

from model.interface import *
# from model.batch_utils import batch_coords, batch_Abattens
from training.train_utils import *
from utils.general import exists, contains_non_empty_string

ATOM_DIM = 3


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adjacency_matrix):
        # support = torch.mm(x, self.weight)
        # output = torch.mm(adjacency_matrix, support) + self.bias
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adjacency_matrix, support) + self.bias
        return output


class GraphEmbeddingGCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GraphEmbeddingGCN, self).__init__()
        self.layer_norm1 = nn.LayerNorm(input_size)
        self.gc1 = GraphConvolution(input_size, hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.gc2 = GraphConvolution(hidden_size, output_size)

    def forward(self, x, adjacency_matrix):
        x = self.layer_norm1(x)
        x = F.relu(self.gc1(x, adjacency_matrix))
        x = self.layer_norm2(x)
        x = self.gc2(x, adjacency_matrix)
        return x


class FAbAtInterGCN(pl.LightningModule):
    def __init__(
        self,
        config,
        config_overwrite=None,
        inter_criterion=None,
        trunc_distance=10,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.inter_criterion = inter_criterion
        config = self.hparams.config
        self.predcoords_exist = config.predcoords_exist if 'predcoords_exist' in dir(config) else False
        self.only_muterange = config.only_muterange if 'only_muterange' in dir(config) else False
        self.only_cdr = config.only_cdr if 'only_cdr' in dir(config) else False
        self.finturn = True if contains_non_empty_string(config.synergy_finetunelayers) else False
        if not isinstance(config, dict):  config = vars(config)
        if exists(config_overwrite):
            config.update(config_overwrite)

        if 'node_dim' in config:
            self.node_dim = config['node_dim']
            if self.node_dim <= 22:
                self.esm_feat_dim = 11
            else:
                self.esm_feat_dim = 36
        else:
            self.esm_feat_dim = 36
            self.node_dim = 36
        self.at_node_transform = nn.Sequential(
            nn.Linear(self.esm_feat_dim, self.node_dim,),
            nn.ReLU(),
            nn.LayerNorm(self.node_dim),
        )
        # self.At_Graph_embeddinglayer = GraphEmbeddingGCN(self.esm_feat_dim, 2*self.esm_feat_dim, self.esm_feat_dim)
        # self.Ab_Graph_embeddinglayer = GraphEmbeddingGCN(self.esm_feat_dim, 2*self.esm_feat_dim, self.esm_feat_dim)
        # self.graphout_normls = nn.ModuleList([nn.LayerNorm(self.esm_feat_dim), nn.LayerNorm(self.esm_feat_dim)])
        # self.class_layer = nn.Sequential(nn.Linear(self.esm_feat_dim * 2, self.esm_feat_dim), nn.ReLU(), nn.Linear(self.esm_feat_dim, 2))
        self.At_Graph_embeddinglayer = GraphEmbeddingGCN(self.node_dim, 2*self.node_dim, self.node_dim)
        self.Ab_Graph_embeddinglayer = GraphEmbeddingGCN(self.node_dim, 2*self.node_dim, self.node_dim)
        self.graphout_normls = nn.ModuleList([nn.LayerNorm(self.node_dim), nn.LayerNorm(self.node_dim)])
        self.class_layer = nn.Sequential(nn.Linear(self.node_dim * 2, self.node_dim), nn.ReLU(), nn.Linear(self.node_dim, 2))


    def forward(
        self,
        data: SingleDataSamp,
        batch_size = 1,
        inter_loss_weight = 1.0,
    ):
        flat_coordsi = data.Ab_pred_coords[:, :, 0, :].squeeze()
        At_embeddings, At_coords_label, inter_label = data.At_embeddings, data.At_coords_label, data.inter_label
        At_coords_label = At_coords_label[:, :, 0, :].squeeze()
        Ab_phisicEmbed = normalize_features(data.Ab_phisicEmbed)
        At_embeddings = normalize_features(At_embeddings)
        if flat_coordsi.dim() == 2:  flat_coordsi = flat_coordsi.unsqueeze(0)
        if At_coords_label.dim() == 2:  At_coords_label = At_coords_label.unsqueeze(0)

        if self.esm_feat_dim == 11:
            At_embeddings = At_embeddings[:, :, 25:]
            Ab_phisicEmbed = Ab_phisicEmbed[:, :, 25:]
        else:
            At_embeddings = At_embeddings[:, :, :]
            Ab_phisicEmbed = Ab_phisicEmbed[:, :, :]

        at_str_nodes = self.at_node_transform(At_embeddings)
        at_adj = generate_adjacency_matrix(At_coords_label)
        ab_str_nodes = self.at_node_transform(Ab_phisicEmbed)
        ab_adj = generate_adjacency_matrix(flat_coordsi)

        # at_str_edges = self.at_edge_transform(At_attentions)W
        # at_str_nodes, at_str_edges = self.At_main_block(at_nodes_mat, at_edges_mat, mask=At_res_batch_mask)
        ab_str_nodes = self.At_Graph_embeddinglayer(ab_str_nodes, ab_adj)
        at_str_nodes = self.Ab_Graph_embeddinglayer(at_str_nodes, at_adj)
        ab_str_nodes = self.graphout_normls[0](ab_str_nodes)
        at_str_nodes = self.graphout_normls[1](at_str_nodes)

        ab_str_nodes = ab_str_nodes.mean(axis=1)
        at_str_nodes = at_str_nodes.mean(axis=1)
        inter_feat = torch.cat([ab_str_nodes, at_str_nodes], axis=1)
        inter_prob = self.class_layer(inter_feat)
        inter_loss = inter_class_loss(inter_prob, inter_label, self.inter_criterion, loss_weight=inter_loss_weight)

        output = InterFoldOutput(
            coords=None,
            prmsd=None,
            translations=None,
            rotations=None,
            coords_loss=None,
            bondlen_loss=None,
            prmsd_loss=None,
            inter_loss=inter_loss,
            inter_prob=inter_prob.detach(),
            loss=inter_loss,
        )
        return output


def generate_adjacency_matrix(coordinates, threshold=8.0):
    # coordinates: [batch_size, num_atoms, 3]

    # 将原子坐标都为原点的情况排除在外
    is_not_origin = torch.any(coordinates != 0, dim=-1)
    # 扩展 is_not_origin 维度以便正确计算距离
    is_not_origin = is_not_origin[:, :, None]

    # 计算欧几里得距离，只对非原点的原子计算距离
    distances = torch.norm(coordinates[:, :, None, :] - coordinates[:, None, :, :], dim=-1) * is_not_origin * is_not_origin.transpose(1, 2)
    # 生成邻接矩阵，距离小于阈值的为1，否则为0
    adjacency_matrix = (distances < threshold).float()
    # 将自身到自身的连接设为0
    adjacency_matrix = adjacency_matrix * (1 - torch.eye(coordinates.size(1), device=coordinates.device)[None, :, :])
    return adjacency_matrix


def normalize_features(features, axis=2):
    # 计算均值和标准差
    mean_vals = torch.mean(features, dim=axis, keepdim=True)
    std_devs = torch.std(features, dim=axis, keepdim=True)
    # 归一化
    normalized_features = (features - mean_vals) / (std_devs + 1e-8)  # 添加一个小的常数以避免除零错误
    return normalized_features


if __name__ == "__main__":
    # 生成一个小规模的图数据作为示例
    adjacency_matrix = torch.tensor([[0, 1, 0, 1],
                                    [1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [1, 0, 1, 0]], dtype=torch.float32)

    # 特征矩阵，每个节点一个特征
    feature_matrix = torch.eye(4)

    # 初始化图嵌入GCN模型
    embedding_model = GraphEmbeddingGCN(input_size=4, hidden_size=2, output_size=2)

    # 优化器
    optimizer = optim.Adam(embedding_model.parameters(), lr=0.01)

    # 训练模型
    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        embedding_output = embedding_model(feature_matrix, adjacency_matrix)
        
        # 对图嵌入向量进行分类，这里使用简单的全连接层
        classification_layer = nn.Linear(2, 2)
        classification_output = classification_layer(embedding_output)
        
        labels = torch.tensor([0, 1], dtype=torch.long)  # 根据实际情况修改标签
        loss = F.cross_entropy(classification_output, labels)
        
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')
