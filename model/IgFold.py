import os
from einops import rearrange, repeat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from model.interface import *
from model.components import TriangleGraphTransformer, IPAEncoder, IPATransformer
from utils.coordinates import get_ideal_coords, place_o_coords
from utils.transforms import quaternion_to_matrix
from training.train_utils import *
from utils.general import exists

ATOM_DIM = 3


class IgFold(pl.LightningModule):

    def __init__(
        self,
        config,
        config_overwrite=None,
    ):
        super().__init__()

        self.save_hyperparameters()
        config = self.hparams.config
        if not isinstance(config, dict):  config = vars(config)
        if exists(config_overwrite):
            config.update(config_overwrite)

        self.bert_feat_dim = 512
        self.bert_attn_dim = 64

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

        self.str_node_transform = nn.Sequential(
            nn.Linear(
                self.bert_feat_dim,
                self.node_dim,
            ),
            nn.ReLU(),
            nn.LayerNorm(self.node_dim),
        )
        self.str_edge_transform = nn.Sequential(
            nn.Linear(
                self.bert_attn_dim,
                self.node_dim,
            ),
            nn.ReLU(),
            nn.LayerNorm(self.node_dim),
        )

        self.main_block = TriangleGraphTransformer(
            dim=self.node_dim,
            edge_dim=self.node_dim,
            depth=self.depth,
            tri_dim_hidden=2 * self.node_dim,
            gt_depth=self.gt_depth,
            gt_heads=self.gt_heads,
            gt_dim_head=self.node_dim // 2,
        )
        self.template_ipa = IPAEncoder(
            dim=self.node_dim,
            depth=self.temp_ipa_depth,
            heads=self.temp_ipa_heads,
            require_pairwise_repr=True,
        )  # 将template的(rotation, translation)、目标序列的pair_rep(边信息)Encode到目标序列的node_feat中，并输出更新后的node_feat

        self.structure_ipa = IPATransformer(
            dim=self.node_dim,
            depth=self.str_ipa_depth,
            heads=self.str_ipa_heads,
            require_pairwise_repr=True,
        )  # 采用了3层IPA

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
        self.dev_linear = nn.Linear(
            self.node_dim,
            4,
        )

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

    def clean_input(
        self,
        input: IgFoldInput,
    ):
        embeddings = input.embeddings
        temp_coords = input.template_coords
        temp_mask = input.template_mask
        batch_mask = input.batch_mask
        align_mask = input.align_mask

        batch_size = embeddings[0].shape[0]  # batch_size等于抗体H链的embedding dim 0的维度
        seq_lens = [max(e.shape[1], 0) for e in embeddings]
        seq_len = sum(seq_lens)

        if not exists(temp_coords):
            temp_coords = torch.zeros(
                batch_size,
                4 * seq_len,
                ATOM_DIM,
                device=self.device,
            ).float()
        if not exists(temp_mask):
            temp_mask = torch.zeros(
                batch_size,
                4 * seq_len,
                device=self.device,
            ).bool()
        if not exists(batch_mask):
            batch_mask = torch.ones(
                batch_size,
                4 * seq_len,
                device=self.device,
            ).bool()
        if not exists(align_mask):
            align_mask = torch.ones(
                batch_size,
                4 * seq_len,
                device=self.device,
            ).bool()

        align_mask = align_mask & batch_mask  # Should already be masked by batch_mask anyway
        temp_coords[~temp_mask] = 0.
        for i, (tc, m) in enumerate(zip(temp_coords, temp_mask)):
            temp_coords[i][m] -= tc[m].mean(-2)

        input.template_coords = temp_coords
        input.template_mask = temp_mask
        input.batch_mask = batch_mask
        input.align_mask = align_mask

        return input, batch_size, seq_lens, seq_len

    def forward(
        self,
        input: IgFoldInput,
    ):
        input, batch_size, seq_lens, seq_len = self.clean_input(input)
        embeddings = input.embeddings  # 输入是一个列表, [embed_l1, embed_l2]， 其中embed_i大小为 [batch, ri, cm], 目前c_m为512, ri为氨基酸序列长度
        attentions = input.attentions  # 输入是一个列表, [attn_l1, attn_l2]， 其中attn_l1大小为 [batch, 8, 8, ri, ri]
        temp_coords = input.template_coords  # 输入是一个tensor, [batch, sum(ri)*4, 3]， 含义是4个主原子的模板结构坐标
        temp_mask = input.template_mask  # 输入是一个tensor, 大小为[batch, sum(ri)*4]， 含义是哪些氨基酸索引的temp_coords是可用的
        coords_label = input.coords_label  # tensor， [batch, sum(ri), 5, 3]， 含义是每个氨基酸上的5种重原子：[N, CA, C, CB, O]的坐标矩阵
        batch_mask = input.batch_mask  # tensor, [batch, max(sum(ri))], 当batch不为1时，由于先前对feat和edges数据做补齐，此处作为多条序列的氨基酸索引数据的mask
        align_mask = input.align_mask
        return_embeddings = input.return_embeddings  # bool, 输出时是否再返回embeddings

        res_batch_mask = rearrange(batch_mask, "b (l a) -> b l a", a=4,).all(-1).to(self.device)  # torch.all(input, dim) 对于给定dim中的每一行，若全为True则返回True, 否则返回False
        res_temp_mask = rearrange(temp_mask, "b (l a) -> b l a", a=4,).all(-1).to(self.device)

        ### Model forward pass

        bert_feats = torch.cat(embeddings, dim=1).to(self.device)  # [batch, ri, ri, cm']，对应的是AlphaFold2中的pair_rep特征
        bert_attn = torch.zeros(
            (batch_size, seq_len, seq_len, self.bert_attn_dim),
            device=self.device,
        )  # [b, ri, ri, cz'] cz'代表attn维度的向量维数，此处为64
        for i, (a, l) in enumerate(zip(attentions, seq_lens)):  # 得到bert预训练模型的attn注意力矩阵
            a = rearrange(a, "b n h l1 l2 -> b l1 l2 (n h)")  # n代表bert中每一层的attn_score，h代表每层中attn过程的heads数目
            cum_l = sum(seq_lens[:i])
            bert_attn[:, cum_l:cum_l + l, cum_l:cum_l + l, :] = a

        temp_translations, temp_rotations = self.get_coords_tran_rot(
            temp_coords,
            batch_size,
            seq_len,
        )  # 利用res_ideal_coords(这里是black hole初始化返回的重原子坐标)和temp，得到一组旋转矩阵和偏移向量

        str_nodes = self.str_node_transform(bert_feats)  # [batch, ri, cm] 氨基酸特征进行特征映射，将cm 512-->64
        str_edges = self.str_edge_transform(bert_attn)  # [batch, ri, ri, cz] edge信息进行特征映射，使得两者特征维度相同  将cz  64-->64

        # (alphafold中的EvoFormer还将node得到的attn结果利用外积与edges隐含层特征建立了联系，在此处没有这么做，只不过node加入了pair_rep_bias)
        str_nodes, str_edges = self.main_block(
            str_nodes,
            str_edges,
            mask=res_batch_mask,
        )  # 利用三角形乘法以及"incoming"+"outgoing" attn计算对nodes和edges特征进行更新

        # 将template的(rotation, translation)、目标序列的pair_rep(边信息)Encode到目标序列的node_feat中，并输出更新后的node_feat
        gt_embs = str_nodes
        str_nodes = self.template_ipa(
            str_nodes,
            translations=temp_translations,
            rotations=temp_rotations,
            pairwise_repr=str_edges,
            mask=res_temp_mask,  # 通过屏蔽不同时具有已知坐标的残基对之间的注意力来合并部分结构信息；因此假设没有模板信息时，template_ipa所实现的只是通过FFN来更新nodes信息
        )  # 利用一个IPA引入模板结构信息，并不对骨架的旋转矩阵以及偏移向量产生更新量
        structure_embs = str_nodes

        ipa_coords, ipa_translations, ipa_quaternions = self.structure_ipa(
            str_nodes,
            translations=None,
            quaternions=None,
            pairwise_repr=str_edges,
            mask=res_batch_mask,
        )  # 输入为node_feat，pair_rep，初始化的Ti，首先基于上述三个输入，通过IPA层对node_feat进行更新，每次IPA层输出的node_feat都经过一个线性层预测当前帧的Ti更新量，最终经过多个循环返回总的Ti更新量
        # 经过上述多个IPA循环后，利用全局的新Ti以及初始化的理想的氨基酸自由浮动骨架，得到预测序列的氨基酸4重原子坐标，以及Ti
        ipa_rotations = quaternion_to_matrix(ipa_quaternions)

        dev_nodes = self.dev_node_transform(bert_feats)
        dev_edges = self.dev_edge_transform(bert_attn)
        dev_out_feats = self.dev_ipa(
            dev_nodes,
            translations=ipa_translations.detach(),
            rotations=ipa_rotations.detach(),
            pairwise_repr=dev_edges,
            mask=res_batch_mask,
        )
        dev_pred = F.relu(self.dev_linear(dev_out_feats))  # [b r 4]
        dev_pred = rearrange(dev_pred, "b l a -> b (l a)", a=4)
        # The coordinate frame for each residue is defined by four atoms 由四类原子定义蛋白质骨架
        bb_coords = rearrange(
            ipa_coords[:, :, :3],
            "b l a d -> b (l a) d",
        )
        flat_coords = rearrange(
            ipa_coords[:, :, :4],
            "b l a d -> b (l a) d",
        )

        ### Calculate losses if given labels
        loss = torch.zeros(
            batch_size,
            device=self.device,
        )
        if exists(coords_label):
            coords_label = rearrange(coords_label[:, :, :4], "b l a d -> b (l a) d")  # 蛋白质4种重原子构成的坐标
            rmsd_clamp = self.hparams.config["rmsd_clamp"]
            coords_loss = kabsch_mse(
                flat_coords,
                coords_label,
                align_mask=batch_mask,
                mask=batch_mask,
                clamp=rmsd_clamp,
            )  # 经过Kabsch对所有预测原子坐标以及label原子坐标对齐后的4种主碳原子距离的均方误差

            bb_coords_label = rearrange(
                rearrange(coords_label, "b (l a) d -> b l a d", a=4)[:, :, :3],
                "b l a d -> b (l a) d"
            )
            bb_batch_mask = rearrange(
                rearrange(batch_mask, "b (l a) -> b l a", a=4)[:, :, :3],
                "b l a -> b (l a)"
            )
            bondlen_loss = bond_length_l1(
                bb_coords,
                bb_coords_label,
                bb_batch_mask,
            )

            prmsd_loss = []
            cum_seq_lens = np.cumsum([0] + seq_lens)
            for sl_i, sl in enumerate(seq_lens):
                align_mask_ = align_mask.clone()
                align_mask_[:, :cum_seq_lens[sl_i]] = False
                align_mask_[:, cum_seq_lens[sl_i + 1]:] = False
                res_batch_mask_ = res_batch_mask.clone()
                res_batch_mask_[:, :cum_seq_lens[sl_i]] = False
                res_batch_mask_[:, cum_seq_lens[sl_i + 1]:] = False

                if sl == 0 or align_mask_.sum() == 0 or res_batch_mask_.sum() == 0:
                    continue

                prmsd_loss.append(  # 这里输入的是dev_pred(也就是以bert_feats为输入经过两层IPAEncoder后的输出)
                    bb_prmsd_l1(
                        dev_pred,
                        flat_coords.detach(),
                        coords_label,
                        align_mask=align_mask_,
                        mask=res_batch_mask_,
                    )
                )
            prmsd_loss = sum(prmsd_loss)

            coords_loss, bondlen_loss = list(
                map(
                    lambda l: rearrange(l, "(c b) -> b c", b=batch_size).
                    mean(1),
                    [coords_loss, bondlen_loss],
                )
            )

            loss += sum([coords_loss, bondlen_loss, prmsd_loss])
        else:
            prmsd_loss, coords_loss, bondlen_loss = None, None, None

        if not exists(coords_label):
            loss = None

        bert_embs = bert_feats if return_embeddings else None
        bert_attn = bert_attn if return_embeddings else None
        gt_embs = gt_embs if return_embeddings else None
        structure_embs = structure_embs if return_embeddings else None
        output = IgFoldOutput(
            coords=ipa_coords,
            prmsd=dev_pred,
            translations=ipa_translations,
            rotations=ipa_rotations,
            coords_loss=coords_loss,
            bondlen_loss=bondlen_loss,
            prmsd_loss=prmsd_loss,
            loss=loss,
            bert_embs=bert_embs,
            bert_attn=bert_attn,
            gt_embs=gt_embs,
            structure_embs=structure_embs,
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
        output: IgFoldOutput,
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
