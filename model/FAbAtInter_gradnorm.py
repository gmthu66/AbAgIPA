"""本版本基于1219版本基础上, 添加互作方面的预测网络"""
from einops import rearrange, repeat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim

from model.interface import *
from model.components import TriangleGraphTransformer, IPAEncoder, IPATransformer, InterEdge, Inter_GlobalMean
from training.train_utils import inter_class_loss
from utils.coordinates import get_ideal_coords, place_o_coords
from utils.transforms import quaternion_to_matrix
from training.train_utils import *
from utils.general import exists

ATOM_DIM = 3


class FAbAtInter(pl.LightningModule):

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
        if not isinstance(config, dict):  config = vars(config)
        if exists(config_overwrite):
            config.update(config_overwrite)

        self.bert_feat_dim = 512
        self.bert_attn_dim = 64
        self.esm_feat_dim = 1280
        self.esm_attn_dim = 660

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

        self.str_node_transform = nn.Sequential(
            nn.Linear(self.bert_feat_dim, self.node_dim,),
            nn.ReLU(),
            nn.LayerNorm(self.node_dim),
        )
        self.str_edge_transform = nn.Sequential(
            nn.Linear(self.bert_attn_dim, self.node_dim,),
            nn.ReLU(),
            nn.LayerNorm(self.node_dim),
        )
        self.at_node_transform = nn.Sequential(
            nn.Linear(self.esm_feat_dim, self.node_dim,),
            nn.ReLU(),
            nn.LayerNorm(self.node_dim),
        )
        self.at_edge_transform = nn.Sequential(
            nn.Linear(self.esm_attn_dim, self.node_dim),
            nn.ReLU(),
            nn.LayerNorm(self.node_dim)
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
        self.At_main_block = TriangleGraphTransformer(
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
        self.At_template_ipa = IPAEncoder(
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
            require_nodes_feat=True,
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
        self.GradNormLoss = GradNormLoss(num_of_task=4)
        self.dev_linear = nn.Linear(self.node_dim, 4,)
        # self.inter_layer = InterAtten(dim=self.node_dim, dim_head=self.node_dim)
        self.inter_layer = InterEdge(depth=self.inter_attn_depth, edge_dim=self.node_dim, dim_head=self.node_dim, heads=4)
        # self.inter_layer = Inter_GlobalMean()
        self.inter_class = nn.Linear(self.node_dim, 2)

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
    ):
        seq_lens, Ab_VHLlens, max_abseqlen = [data.Ab_seqlen], data.Ab_VHLlen, data.Ab_maxlen
        Ab_embeddings = data.Ab_embedding  # 输入是一个列表, [embed_l1, embed_l2]， 其中embed_i大小为 [batch, ri, cm], 目前c_m为512, ri为氨基酸序列长度
        Ab_attentions = data.Ab_attention  # 输入是一个列表, [attn_l1, attn_l2]， 其中attn_l1大小为 [batch, 8, 8, ri, ri]
        Ab_temp_coords = data.Ab_temp_coords  # 输入是一个tensor, [batch, sum(ri)*4, 3]， 含义是4个主原子的模板结构坐标
        Ab_temp_mask = data.Ab_temp_mask  # 输入是一个tensor, 大小为[batch, sum(ri)*4]， 含义是哪些氨基酸索引的temp_coords是可用的
        Ab_coords_label = data.Ab_coords_label  # tensor， [batch, sum(ri), 5, 3]， 含义是每个氨基酸上的5种重原子：[N, CA, C, CB, O]的坐标矩阵
        Ab_batch_mask = data.Ab_batch_mask  # tensor, [batch, max(sum(ri))], 当batch不为1时，由于先前对feat和edges数据做补齐，此处作为多条序列的氨基酸索引数据的mask
        Ab_align_mask = data.Ab_align_mask
        Ab_cdr_mask = data.Ab_cdr_indexs
        # Ab_cdr_indexs = torch.tensor(data.Ab_cdr_indexs, dtype=torch.long, device=self.device) if data.Ab_cdr_indexs is not None else data.Ab_cdr_indexs
        return_embeddings = data.return_embeddings  # bool, 输出时是否再返回embeddings

        res_batch_mask = rearrange(Ab_batch_mask, "b (l a) -> b l a", a=4,).all(-1).to(self.device)  # torch.all(input, dim) 对于给定dim中的每一行，若全为True则返回True, 否则返回False
        # Inter_Ab_res_mask = torch.zeros_like(res_batch_mask).index_fill(dim=1, index=Ab_cdr_indexs, value=1) if Ab_cdr_indexs is not None else res_batch_mask
        res_temp_mask = rearrange(Ab_temp_mask, "b (l a) -> b l a", a=4,).all(-1).to(self.device)
        At_res_batch_mask = rearrange(data.At_batch_mask, "b (l a) -> b l a", a=4).all(-1).to(self.device)

        At_embeddings, At_attentions, At_coords_label, inter_label = data.At_embeddings, data.At_attentions, data.At_coords_label, data.inter_label
        At_coords_label = rearrange(At_coords_label[:, :, :4, :], 'b l a d -> b (l a) d', a=4, d=3)
        ### Model forward pass

        bert_feats = torch.cat(Ab_embeddings, dim=1).to(self.device) if isinstance(Ab_embeddings, list) else Ab_embeddings  # [batch, ri, ri, cm']，对应的是AlphaFold2中的pair_rep特征
        if isinstance(Ab_attentions, list):
            bert_attn = torch.zeros(
                (batch_size, max_abseqlen, max_abseqlen, self.bert_attn_dim),
                device=self.device,
            )  # [b, ri, ri, cz'] cz'代表attn维度的向量维数，此处为64
            for i, (a, l) in enumerate(zip(Ab_attentions, seq_lens)):  # 得到bert预训练模型的attn注意力矩阵
                a = rearrange(a, "b n h l1 l2 -> b l1 l2 (n h)")  # n代表bert中每一层的attn_score，h代表每层中attn过程的heads数目
                cum_l = sum(seq_lens[:i])
                bert_attn[:, cum_l:cum_l + l, cum_l:cum_l + l, :] = a
        else:
            bert_attn = Ab_attentions

        # 利用res_ideal_coords(这里是black hole初始化返回的重原子坐标)和temp，得到一组旋转矩阵和偏移向量
        temp_translations, temp_rotations = self.get_coords_tran_rot(Ab_temp_coords, batch_size, max_abseqlen,)
        At_truth_translations, At_truth_rotations = self.get_coords_tran_rot(At_coords_label, batch_size, At_embeddings.size(1),)

        str_nodes_mat = self.str_node_transform(bert_feats)  # [batch, ri, cm] 氨基酸特征进行特征映射，将cm 512-->64
        str_edges_mat = self.str_edge_transform(bert_attn)  # [batch, ri, ri, cz] edge信息进行特征映射，使得两者特征维度相同  将cz  64-->64
        at_nodes_mat = self.at_node_transform(At_embeddings)
        at_edges_mat = self.at_edge_transform(At_attentions)

        loss = torch.zeros(batch_size, device=self.device,)
        # for i_ in range(batch_size):
        # (alphafold中的EvoFormer还将node得到的attn结果利用外积与edges隐含层特征建立了联系，在此处没有这么做，只不过node加入了pair_rep_bias)
        str_nodes, str_edges = self.main_block(str_nodes_mat, str_edges_mat, mask=res_batch_mask,)  # 利用三角形乘法以及"incoming"+"outgoing" attn计算对nodes和edges特征进行更新
        at_str_nodes, at_str_edges = self.At_main_block(at_nodes_mat, at_edges_mat, mask=At_res_batch_mask)
        at_str_nodes = self.At_template_ipa(at_str_nodes, translations=At_truth_translations, rotations=At_truth_rotations, pairwise_repr=at_str_edges, mask=At_res_batch_mask)

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

        ipa_coords, ipa_translations, ipa_quaternions, str_nodes = self.structure_ipa(
            str_nodes,
            translations=None,
            quaternions=None,
            pairwise_repr=str_edges,
            mask=res_batch_mask,
        )  # 输入为node_feat，pair_rep，初始化的Ti，首先基于上述三个输入，通过IPA层对node_feat进行更新，每次IPA层输出的node_feat都经过一个线性层预测当前帧的Ti更新量，最终经过多个循环返回总的Ti更新量
        assert torch.isnan(ipa_coords).any().item() is False  # 防止预测得到的结构主原子坐标中含有nan值

        inter_edges = rearrange(str_nodes, 'b j d -> b j () d') + rearrange(at_str_nodes, 'b i d -> b () i d')
        inter_feat = self.inter_layer(inter_edges, ab_mask=Ab_cdr_mask, at_mask=At_res_batch_mask)
        inter_feat = inter_feat.squeeze(-1).squeeze(-1)

        # inter_feat = self.inter_layer(str_nodes, at_str_nodes, ab_mask=Ab_cdr_mask, at_mask=At_res_batch_mask)
        inter_prob = self.inter_class(inter_feat)
        inter_loss = inter_class_loss(inter_prob, inter_label, self.inter_criterion, loss_weight=10.0)
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
        flat_coordsi = rearrange(ipa_coords[:, :, :4], "b l a d -> b (l a) d",)

        ### Calculate losses if given labels (每次只计算一个, 原因是会显存溢出) ----  以下代码主要计算样本的loss
        if exists(Ab_coords_label):
            # Ab_coords_labeli = Ab_coords_label
            Ab_coords_label = rearrange(Ab_coords_label[:, :, :4], "b l a d -> b (l a) d")  # 蛋白质4种重原子构成的坐标
            # rmsd_clamp = self.hparams.config["rmsd_clamp"]
            rmsd_clamp = getattr(self.hparams.config, "rmsd_clamp")
            coords_loss = kabsch_mse(
                flat_coordsi,
                Ab_coords_label,
                align_mask=Ab_batch_mask,
                mask=Ab_batch_mask,
                clamp=rmsd_clamp,
            )  # 经过Kabsch对所有预测原子坐标以及label原子坐标对齐后的4种主碳原子距离的均方误差

            bb_coords_label = rearrange(
                rearrange(Ab_coords_label, "b (l a) d -> b l a d", a=4)[:, :, :3],
                "b l a d -> b (l a) d"
            )
            bb_batch_mask = rearrange(
                rearrange(Ab_batch_mask, "b (l a) -> b l a", a=4)[:, :, :3],
                "b l a -> b (l a)"
            )
            bondlen_loss = bond_length_l1(
                bb_coords,
                bb_coords_label,
                bb_batch_mask,
            )

            prmsd_loss, seq_lensi = [], Ab_VHLlens
            cum_seq_lens = np.cumsum([0] + seq_lensi)
            # 获取prmsd_loss
            for sl_i, sl in enumerate(seq_lensi):
                align_mask_ = Ab_align_mask.clone()
                align_mask_[:, :cum_seq_lens[sl_i]] = False
                align_mask_[:, cum_seq_lens[sl_i + 1]:] = False
                res_batch_mask_ = res_batch_mask.clone()
                res_batch_mask_[:, :cum_seq_lens[sl_i]] = False
                res_batch_mask_[:, cum_seq_lens[sl_i + 1]:] = False

                if sl == 0 or align_mask_.sum() == 0 or res_batch_mask_.sum() == 0:
                    continue

                prmsd_loss.append(  # 这里输入的是dev_pred(也就是以bert_feats为输入经过两层IPAEncoder后的输出)
                    bb_prmsd_l1(
                        dev_pred,  # 针对residue误差的预测
                        flat_coordsi.detach(),  # 预测得到的三维坐标
                        Ab_coords_label,  # 实际三维坐标标签
                        align_mask=align_mask_,
                        mask=res_batch_mask_,
                    )
                )
            prmsd_loss = sum(prmsd_loss)

            coords_loss, bondlen_loss = list(
                map(
                    lambda l: rearrange(l, "(c b) -> b c", b=1).mean(1),
                    [coords_loss, bondlen_loss],
                )
            )

            loss += sum([coords_loss, bondlen_loss, prmsd_loss, inter_loss])
        else:
            prmsd_loss, coords_loss, bondlen_loss, inter_loss = None, None, None, None

        if not exists(Ab_coords_label):
            loss = None

        bert_embs = bert_feats if return_embeddings else None
        bert_attn = bert_attn if return_embeddings else None
        gt_embs = gt_embs if return_embeddings else None
        structure_embs = structure_embs if return_embeddings else None
        output = InterFoldOutput(
            coords=ipa_coords,
            prmsd=dev_pred,
            translations=ipa_translations,
            rotations=ipa_rotations,
            coords_loss=coords_loss,
            bondlen_loss=bondlen_loss,
            prmsd_loss=prmsd_loss,
            inter_loss=inter_loss,
            inter_prob=inter_prob.detach(),
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


class GradNormLoss(nn.Module):
    def __init__(self, num_of_task, alpha=1.5):
        super(GradNormLoss, self).__init__()
        self.num_of_task = num_of_task
        self.alpha = alpha
        self.w = nn.Parameter(torch.ones(num_of_task, dtype=torch.float))
        self.l1_loss = nn.L1Loss()
        self.L_0 = None

    def forward(self, L_t: torch.Tensor):
        # initialize the initial loss `Li_0`
        if self.L_0 is None:
            self.L_0 = L_t.detach() # detach
        # compute the weighted loss w_i(t) * L_i(t)
        self.L_t = L_t
        self.wL_t = L_t * self.w
        # the reduced weighted loss
        self.total_loss = self.wL_t.sum()
        return self.total_loss        

    # additional forward & backward pass
    def additional_forward_and_backward(self, grad_norm_weights: nn.Module, optimizer: optim.Optimizer):
        # do `optimizer.zero_grad()` outside
        self.total_loss.backward(retain_graph=True)
        # in standard backward pass, `w` does not require grad
        self.w.grad.data = self.w.grad.data * 0.0

        self.GW_t = []
        for i in range(self.num_of_task):
            # get the gradient of this task loss with respect to the shared parameters
            GiW_t = torch.autograd.grad(
                self.L_t[i], grad_norm_weights.parameters(),
                    retain_graph=True, create_graph=True)
            # compute the norm
            self.GW_t.append(torch.norm(GiW_t[0] * self.w[i]))
        self.GW_t = torch.stack(self.GW_t) # do not detatch
        self.bar_GW_t = self.GW_t.detach().mean()
        self.tilde_L_t = (self.L_t / self.L_0).detach()
        self.r_t = self.tilde_L_t / self.tilde_L_t.mean()
        grad_loss = self.l1_loss(self.GW_t, self.bar_GW_t * (self.r_t ** self.alpha))
        self.w.grad = torch.autograd.grad(grad_loss, self.w)[0]
        optimizer.step()

        self.GW_ti, self.bar_GW_t, self.tilde_L_t, self.r_t, self.L_t, self.wL_t = None, None, None, None, None, None
        # re-norm
        self.w.data = self.w.data / self.w.data.sum() * self.num_of_task


# This is AN interface.
class GradNormModel:
    def get_grad_norm_weights(self) -> nn.Module:
        raise NotImplementedError(
            "Please implement the method `get_grad_norm_weights`")
