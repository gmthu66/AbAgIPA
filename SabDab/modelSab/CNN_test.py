"""本版本基于1219版本基础上, 添加互作方面的预测网络"""
from einops import rearrange, repeat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim

from copy import deepcopy
from model.interface import *
from model.components import TriangleGraphTransformer, IPAEncoder, IPATransformer, Inter_GlobalMean, BinaryClassifier, My1DCNN
# from model.batch_utils import batch_coords, batch_Abattens
from utils.coordinates import get_ideal_coords, place_o_coords
from utils.transforms import quaternion_to_matrix
from training.train_utils import *
from utils.general import exists, contains_non_empty_string

ATOM_DIM = 3


class FAbAtInterPhysic(pl.LightningModule):

    def __init__(
        self,
        config,
        config_overwrite=None,
        inter_criterion=None,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.inter_criterion = inter_criterion
        config = self.hparams.config
        self.only_muterange = config.only_muterange if 'only_muterange' in dir(config) else False
        self.only_cdr = config.only_cdr if 'only_cdr' in dir(config) else False
        self.finturn = True if contains_non_empty_string(config.synergy_finetunelayers) else False
        if not isinstance(config, dict):  config = vars(config)
        if exists(config_overwrite):
            config.update(config_overwrite)

        self.bert_feat_dim = 512
        self.bert_attn_dim = 64
        self.esm_feat_dim = 36
        self.esm_attn_dim = 72

        self.node_dim = config["node_dim"]

        self.depth = config["depth"]
        self.gt_depth = config["gt_depth"]
        self.gt_heads = config["gt_heads"]

        self.temp_ipa_depth = config["temp_ipa_depth"]  # 引入template旋转矩阵/偏移向量信息的 IPA Layer所使用的网络超参数（对feat_node特征矩阵进行更新）
        self.temp_ipa_heads = config["temp_ipa_heads"]

        self.str_ipa_depth = config["str_ipa_depth"]
        self.str_ipa_heads = config["str_ipa_heads"]

        self.dev_ipa_depth = config["dev_ipa_depth"]
        self.dev_ipa_heads = config["dev_ipa_heads"]

        self.inter_attn_depth = config["inter_attn_depth"]

        # self.At_mainBLK_depth = getattr(config, "At_depth") if hasattr(config, "At_depth") else 1
        self.At_temp_ipa_depth = getattr(config, "At_temp_ipa_depth") if hasattr (config, "At_temp_ipa_depth") else 1

        self.ab_node_transform = My1DCNN(self.esm_feat_dim, self.node_dim, kernel_size=3)
        self.at_node_transform = My1DCNN(self.esm_feat_dim, self.node_dim, kernel_size=3)
        # self.ab_node_transform = nn.Sequential(
        #     nn.Linear(self.esm_feat_dim, self.node_dim,),
        #     nn.ReLU(),
        #     nn.LayerNorm(self.node_dim),
        # )
        # self.ab_edge_transform = nn.Sequential(
        #     nn.Linear(self.esm_attn_dim, self.node_dim),
        #     nn.ReLU(),
        #     nn.LayerNorm(self.node_dim)
        # )
        # self.at_node_transform = nn.Sequential(
        #     nn.Linear(self.esm_feat_dim, self.node_dim,),
        #     nn.ReLU(),
        #     nn.LayerNorm(self.node_dim),
        # )
        # self.at_edge_transform = nn.Sequential(
        #     nn.Linear(self.esm_attn_dim, self.node_dim),
        #     nn.ReLU(),
        #     nn.LayerNorm(self.node_dim)
        # )

        # self.At_main_block = TriangleGraphTransformer(
        #     dim=self.node_dim,
        #     edge_dim=self.node_dim,
        #     depth=self.At_mainBLK_depth,
        #     tri_dim_hidden=2 * self.node_dim,
        #     gt_depth=self.gt_depth,
        #     gt_heads=self.gt_heads,
        #     gt_dim_head=self.node_dim // 2,
        # )
        # 将template的(rotation, translation)、目标序列的pair_rep(边信息)Encode到目标序列的node_feat中，并输出更新后的node_feat

        self.dev_node_transform = nn.Sequential(
            nn.Linear(self.bert_feat_dim, self.node_dim),
            nn.ReLU(),
            nn.LayerNorm(self.node_dim),
        )

        self.dev_edge_transform = nn.Sequential(
            nn.Linear(self.bert_attn_dim, self.node_dim,),
            nn.ReLU(),
            nn.LayerNorm(self.node_dim),
        )
        self.dev_ipa = IPAEncoder(
            dim=self.node_dim,
            depth=self.dev_ipa_depth,
            heads=self.dev_ipa_heads,
            require_pairwise_repr=True,
        )
        self.dev_linear = nn.Linear(self.node_dim, 4,)
        self.inter_layer = Inter1dCNN(self.node_dim, self.node_dim)
        self.inter_class = BinaryClassifier(self.node_dim, self.node_dim * 2)
        # self.inter_class = nn.Linear(self.node_dim * 2, 2)

    def get_coords_tran_rot(
        self,
        temp_coords,
        batch_size,
        seq_len,
    ):
        res_coords = rearrange(
            temp_coords,
            "b (l a) d -> b l a d",
            l=seq_len,
        ).to(self.device)
        ideal_coords = get_ideal_coords()
        res_ideal_coords = repeat(
            ideal_coords,
            "a d -> b l a d",
            b=batch_size,
            l=seq_len,
        ).to(self.device)  # 利用res_ideal_coords(这里是black hole初始化返回的重原子坐标)和temp，得到一组旋转矩阵和偏移向量
        _, rotations, translations = kabsch(
            res_ideal_coords,
            res_coords,
            return_translation_rotation=True,
        )  # translations 偏移向量构成的tensor：[batch, ri, 1,3]; rotations 旋转矩阵构成的tensor： [batch, ri, 3, 3]
        translations = rearrange(
            translations,
            "b l () d -> b l d",
        )  # rearrange起到了squeeze（删除空维度）的效果

        return translations, rotations

    def forward(
        self,
        data: SingleDataSamp,
        batch_size = 1,
        inter_loss_weight = 1.0,
    ):
        seq_lens = [data.Ab_seqlen] if not isinstance(data.Ab_seqlen, list) else data.Ab_seqlen
        Ab_embeddings = data.Ab_embedding  # 输入是一个列表, [embed_l1, embed_l2]， 其中embed_i大小为 [batch, ri, cm], 目前c_m为512, ri为氨基酸序列长度
        Ab_VHLlens, max_abseqlen, batch_size = data.Ab_VHLlen, data.Ab_maxlen, Ab_embeddings.size(0)
        Ab_attentions = data.Ab_attention  # 输入是一个列表, [attn_l1, attn_l2]， 其中attn_l1大小为 [batch, 8, 8, ri, ri]
        Ab_temp_coords = data.Ab_temp_coords  # 输入是一个tensor, [batch, sum(ri)*4, 3]， 含义是4个主原子的模板结构坐标
        Ab_temp_mask = data.Ab_temp_mask  # 输入是一个tensor, 大小为[batch, sum(ri)*4]， 含义是哪些氨基酸索引的temp_coords是可用的
        Ab_coords_label = data.Ab_coords_label  # tensor， [batch, sum(ri), 5, 3]， 含义是每个氨基酸上的5种重原子：[N, CA, C, CB, O]的坐标矩阵
        Ab_batch_mask = data.Ab_batch_mask  # tensor, [batch, max(sum(ri))], 当batch不为1时，由于先前对feat和edges数据做补齐，此处作为多条序列的氨基酸索引数据的mask
        Ab_align_mask = data.Ab_align_mask

        return_embeddings = data.return_embeddings  # bool, 输出时是否再返回embeddings
        inter_loss_weight = 1.0 if ((not exists(Ab_coords_label)) or (exists(data.Ab_mut_range))) or (not self.finturn) else 10.0

        res_batch_mask = rearrange(Ab_batch_mask, "b (l a) -> b l a", a=4,).all(-1).to(self.device)  # torch.all(input, dim) 对于给定dim中的每一行，若全为True则返回True, 否则返回False
        res_temp_mask = rearrange(Ab_temp_mask, "b (l a) -> b l a", a=4,).all(-1).to(self.device)
        At_res_batch_mask = rearrange(data.At_batch_mask, "b (l a) -> b l a", a=4).all(-1).to(self.device)

        # ----------------------- 获得理化性质, 进行后续的网络 ----------------------
        if exists(data.Ab_mut_range) and self.only_muterange:
            st, ed = (data.Ab_mut_range[0], data.Ab_mut_range[1]) if (res_temp_mask.size(0) == 1) else (data.Ab_mut_range[0][0], data.Ab_mut_range[0][1])
            Ab_phisicEmbed = data.Ab_phisicEmbed[:, st - 1 : ed + 2, :]
            Ab_phisicAttention = data.Ab_phisicAttention[:, st - 1 : ed + 2, st-1 : ed+2, :]
            Ab_cdr_mask = data.Ab_cdr_indexs[:, st-1 : ed+2]
            Ab_res_batch_mask = res_batch_mask[:, st-1 : ed+2]
        elif self.only_cdr:
            Ab_cdr_mask = data.Ab_cdr_indexs
            Ab_phisicEmbed = data.Ab_phisicEmbed * Ab_cdr_mask.unsqueeze(-1)
            row_Ab_cdr_mask = Ab_cdr_mask.unsqueeze(-1)
            col_Ab_cdr_mask = row_Ab_cdr_mask.transpose(-1, -2)
            row_Ab_cdr_mask = repeat(row_Ab_cdr_mask, 'b l 1 -> b l l2', l2=Ab_cdr_mask.size(1))
            col_Ab_cdr_mask = repeat(col_Ab_cdr_mask, 'b 1 l -> b l2 l', l2=Ab_cdr_mask.size(1))
            Ab_phisicAttention = data.Ab_phisicAttention * (row_Ab_cdr_mask & col_Ab_cdr_mask).unsqueeze(-1)
            Ab_res_batch_mask = Ab_cdr_mask
        else:
            Ab_phisicEmbed, Ab_phisicAttention = data.Ab_phisicEmbed, data.Ab_phisicAttention
            Ab_cdr_mask = data.Ab_cdr_indexs
            Ab_res_batch_mask = res_batch_mask

        At_embeddings, At_attentions, At_coords_label, inter_label = data.At_embeddings, data.At_attentions, data.At_coords_label, data.inter_label
        At_coords_label = rearrange(At_coords_label[:, :, :4, :], 'b l a d -> b (l a) d', a=4, d=3)
        ### Model forward pass

        at_str_nodes = self.at_node_transform(At_embeddings)
        # at_str_edges = self.at_edge_transform(At_attentions)W
        # at_str_nodes, at_str_edges = self.At_main_block(at_nodes_mat, at_edges_mat, mask=At_res_batch_mask)

        ab_str_nodes = self.ab_node_transform(Ab_phisicEmbed)
        # ab_str_edges = self.ab_edge_transform(Ab_phisicAttention)
        # ab_str_nodes, ab_str_edges = self.At_main_block(ab_nodes_mat, ab_edges_mat, mask=Ab_res_batch_mask)

        # if self.only_muterange and exists(data.Ab_mut_range):  # 是否只是保存突变区域的特征, 主要针对HER2的CDRH3
        #     st, ed = (data.Ab_mut_range[0], data.Ab_mut_range[1]) if (res_temp_mask.size(0) == 1) else (data.Ab_mut_range[0][0], data.Ab_mut_range[0][1])
        #     Ab_dumy_translations, Ab_dumy_rotations = self.get_coords_tran_rot(flat_coordsi[:, (st - 1) * 4 : (ed + 2) * 4, :], batch_size, ab_str_nodes.size(1),)
        #     Ab_batch_mask[:, (st - 1) * 4 : (ed + 2) * 4,] = False  #若只是输入突变区域,则对Ab_batch_mask进行更改, 那结构预测loss计算时会将Ab的cdr区域mask掉
        # else:
        #     at_mean_coords = At_coords_label.sum(axis=1) / At_res_batch_mask.sum(axis=1, keepdim=True)
        #     ab_mean_coords = flat_coordsi.sum(axis=1) / Ab_res_batch_mask.sum(axis=1, keepdim=True)
        #     trans_flat_coordsi = flat_coordsi + (at_mean_coords - ab_mean_coords).unsqueeze(1)
        #     Ab_dumy_translations, Ab_dumy_rotations = self.get_coords_tran_rot(trans_flat_coordsi, batch_size, ab_str_nodes.size(1),)
        # ab_str_nodes = self.At_template_ipa(ab_str_nodes, translations=Ab_dumy_translations, rotations=Ab_dumy_rotations, mask=Ab_res_batch_mask)
        # at_str_nodes = self.At_template_ipa(at_str_nodes, translations=At_truth_translations, rotations=At_truth_rotations, mask=At_res_batch_mask)

        # inter_feat = self.inter_layer(ab_str_nodes, at_str_nodes, ab_mask=Ab_cdr_mask, at_mask=At_res_batch_mask)  # 只是在这里进行cdr_mask
        inter_feat = self.inter_layer(ab_str_nodes, at_str_nodes)  # 只是在这里进行cdr_mask
        inter_prob = self.inter_class(inter_feat)
        inter_loss = inter_class_loss(inter_prob, inter_label, self.inter_criterion, loss_weight=inter_loss_weight)

        ### Calculate losses if given labels (每次只计算一个, 原因是会显存溢出) ----  以下代码主要计算样本的loss
        loss = inter_loss
        if not exists(Ab_coords_label) and (not exists(inter_loss)):
            loss = None
        output = InterFoldOutput(
            coords=0,
            prmsd=0,
            translations=0,
            rotations=0,
            inter_loss=inter_loss,
            inter_prob=inter_prob.detach(),
            loss=loss,
            bert_attn=None,
            gt_embs=None,
            structure_embs=None,
        )

        return output

    def score_coords(
        self,
        input: IgFoldInput,
        output: IgFoldOutput,
    ):
        input, _, _, _ = self.clean_input(input)
        batch_mask = input.batch_mask

        res_batch_mask = rearrange(
            batch_mask,
            "b (l a) -> b l a",
            a=4,
        ).all(-1)

        str_translations, str_rotations = output.translations, output.rotations

        bert_feats = output.bert_embs
        bert_attn = output.bert_attn

        dev_nodes = self.dev_node_transform(bert_feats)
        dev_edges = self.dev_edge_transform(bert_attn)
        dev_out_feats = self.dev_ipa(
            dev_nodes,
            translations=str_translations.detach(),
            rotations=str_rotations.detach(),
            pairwise_repr=dev_edges,
            mask=res_batch_mask,
        )
        dev_pred = F.relu(self.dev_linear(dev_out_feats)).squeeze(-1)
        dev_pred = rearrange(dev_pred, "b l a -> b (l a)", a=4)

        return dev_pred

    def transform_ideal_coords(self, translations, rotations):
        b, n, d = translations.shape
        device = translations.device

        ideal_coords = get_ideal_coords().to(device)
        ideal_coords = repeat(
            ideal_coords,
            "a d -> b l a d",
            b=b,
            l=n,
        )
        points_global = torch.einsum(
            'b n a c, b n c d -> b n a d',
            ideal_coords,
            rotations,
        ) + rearrange(
            translations,
            "b l d -> b l () d",
        )

        return points_global

    def gradient_refine(
        self,
        input: IgFoldInput,
        output: InterFoldOutput,
        num_steps: int = 80,
    ):
        input_, _, seq_lens, _ = self.clean_input(input)
        batch_mask = input_.batch_mask
        res_batch_mask = rearrange(
            batch_mask,
            "b (l a) -> b l a",
            a=4,
        ).all(-1)
        translations, rotations = output.translations, output.rotations

        in_coords = self.transform_ideal_coords(translations, rotations).detach()
        in_flat_coords = rearrange(
            in_coords[:, :, :4],
            "b l a d -> b (l a) d",
        )

        with torch.enable_grad():
            translations.requires_grad = True
            rotations.requires_grad = True

            translations = nn.parameter.Parameter(translations)
            rotations = nn.parameter.Parameter(rotations)

            optimizer = torch.optim.Adam([translations, rotations], lr=2e-2)
            for _ in range(num_steps):
                optimizer.zero_grad()

                coords = self.transform_ideal_coords(translations, rotations)
                viol_loss = violation_loss(coords, seq_lens, res_batch_mask)

                flat_coords = rearrange(
                    coords[:, :, :4],
                    "b l a d -> b (l a) d",
                )
                rmsd = kabsch_mse(
                    flat_coords,
                    in_flat_coords,
                    align_mask=batch_mask,
                    mask=batch_mask,
                )  # in_flat_coords是pred

                output.translations = translations
                output.rotations = rotations

                loss = rmsd + viol_loss

                loss.backward()
                optimizer.step()

        prmsd = self.score_coords(input, output)

        coords = place_o_coords(coords)
        output.coords = coords
        output.prmsd = prmsd

        return output


class Inter1dCNN(nn.Module):
    def __init__(self, in_dim, feat_dim, norm_input=True, num_layers=2, rot_dim=4, num_classes=1, num_points=1, point_bias=True, norm='GN', act='leaky_relu', kernel_size=1, dropout=False, num_gn_groups=32):
        super().__init__()
        act_func = nn.LeakyReLU(inplace=True)
        self.layers = nn.ModuleList()
        # self.norm = get_norm(norm, feat_dim, num_gn_groups=num_gn_groups)
        # self.norm = nn.GroupNorm(num_gn_groups, feat_dim)

        if norm_input:
            self.layers.append(nn.BatchNorm1d(in_dim))
        for _i in range(num_layers):
            _indim = in_dim if _i == 1 else feat_dim
            self.layers.append(nn.Conv1d(_indim, feat_dim, kernel_size))
            self.layers.append(nn.GroupNorm(num_gn_groups, feat_dim))
            self.layers.append(act_func)
            if dropout:
                self.layers.append(nn.Dropout(p=0.2))
        # self.layers.append(get_norm(norm, feat_dim, num_gn_groups=num_gn_groups))
        # self.neck = nn.ModuleList()
        # self.neck.append(nn.Conv1d(feat_dim, rot_dim * num_classes, 1))

        self.conv_p = nn.Conv1d(in_channels=feat_dim, out_channels=feat_dim, kernel_size=3, bias=point_bias)

    def forward(self, ab_nodes, at_nodes):
        x = torch.cat([ab_nodes, at_nodes], dim=1).permute(0, 2, 1)  # [b, l1+l2, d]
        for _layer in self.layers:
            x = _layer(x)
        # for _layer in self.neck:
        #     x = _layer(x)

        F.adaptive_avg_pool1d(x, 1)
        # feat = x.clone()
        x = self.conv_p(x)  # 将 feature_dim 移动到第二个维度
        x = F.adaptive_avg_pool1d(x, 1)  # 全局平均池化
        x = x.permute(0, 2, 1)  # 将 feature_dim 移回到最后一个维度

        x = x.squeeze(1)
        x = x.contiguous()

        return x
        # return x, feat
