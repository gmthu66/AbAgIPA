###
#   Inspired by IPA implementation from https://github.com/lucidrains/invariant-point-attention
###

import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from model.components.IPABlock import IPABlock
from utils.coordinates import get_ideal_coords, place_o_coords
from utils.general import exists
from utils.transforms import quaternion_multiply, quaternion_to_matrix


class IPAEncoder(nn.Module):
    # IPA Encoder只是对node_feat进行更新，不对quaternion以及translations产生更新值
    def __init__(
        self,
        *,
        dim,
        depth,
        **kwargs,
    ):
        super().__init__()

        # layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(IPABlock(
                dim=dim,
                **kwargs,
            ))

    def forward(
        self,
        x,  # [b, r, cm]  本质上是single_feat
        *,
        translations=None,
        rotations=None,
        pairwise_repr=None,
        mask=None,
    ):
        for block in self.layers:
            x = block(
                x,
                pairwise_repr=pairwise_repr,
                rotations=rotations,
                translations=translations,
                mask=mask,
            )

        return x


class IPATransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        stop_rotation_grad=False,
        require_nodes_feat=False,
        **kwargs,
    ):
        super().__init__()

        self.stop_rotation_grad = stop_rotation_grad
        self.require_nodes_feat = require_nodes_feat

        self.quaternion_to_matrix = quaternion_to_matrix  # 将一系列四元数转化为 [b, r, 3, 3] 旋转矩阵的函数
        self.quaternion_multiply = quaternion_multiply  # 对两个用四元数形式表示的旋转矩阵进行矩阵乘法的函数

        # layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            ipa_block = IPABlock(
                dim=dim,
                **kwargs,
            )  # IPA block中有 IPA_Attention 以及 FFN
            linear = nn.Linear(dim, 6)
            torch.nn.init.zeros_(linear.weight.data)  # 对Linear层的参数进行初始化
            torch.nn.init.zeros_(linear.bias.data)
            self.layers.append(nn.ModuleList([ipa_block, linear]))

    def forward(
        self,
        single_repr,
        *,
        translations=None,
        quaternions=None,
        pairwise_repr=None,
        mask=None,
    ):
        x, device, quaternion_multiply, quaternion_to_matrix = single_repr, single_repr.device, self.quaternion_multiply, self.quaternion_to_matrix
        b, n, *_ = x.shape

        # if no initial quaternions passed in, start from identity

        if not exists(quaternions):  # 若没有rotations输入则需要进行初始化(每次进入IPATransformer Module时都会对Ti进行重新的初始化)
            quaternions = torch.tensor(
                [1., 0., 0., 0.],
                device=device,
            )  # initial rotations rotations初始化, identity rotation
            quaternions = repeat(
                quaternions,
                'd -> b n d',
                b=b,
                n=n,
            )  # [b, n, d] 代表rotations的四元数Tensor

        # if not translations passed in, start from identity

        if not exists(translations):
            translations = torch.zeros(
                (b, n, 3),
                device=device,
            )  # translations初始化, 用Zero初始化

        # go through the layers and apply invariant point attention and feedforward 进行IPA以及局部每一帧的frame的Ti更新值预测（以及对全局的更新）

        for block, to_update in self.layers:  # self.layers每个元素都是一个Module.list， 含有一个IPA层以及一个Linear层（Linear层weight和bias都被初始化为0）
            rotations = quaternion_to_matrix(quaternions)  # [b, r, 3, 3]
            if self.stop_rotation_grad:
                rotations = rotations.detach()

            x = block(
                x,
                pairwise_repr=pairwise_repr,  # [b, r, r, cz]
                rotations=rotations,  # [b, r, 3, 3]
                translations=translations,  # [b, r, 3]
                mask=mask,
            )  # 点不变注意力层， 利用x也就是feat产生attn_logits_scalar(q_s与k_s基于内积计算i与j之间node的注意力得分)以及attn_logits_point（point与输入的旋转/偏移矩阵相作用，由i与j氨基酸4重原子平均距离dist得到attn_score）, 同时利用pair_rep产生attn_logits_bias
            # 上述3个attn_logits相加得到attn，利用attn对 value_point[b*h, r, 4, 3]、value_scalar[b*h, r, c]以及pair_rep [b*h, r, r, cz] 进行点不变加权求和
            # 返回[b, r, cm]

            # update quaternion and translation
            quaternion_update, translation_update = to_update(x).chunk(
                2,
                dim=-1,
            )  # 四元数固定第一个数字为1，另外3个数字由网络预测得到；此外平移向量为R^3。因此基于single的feat进行每个氨基酸的Ti预测的Linear层参数矩阵为 [cm, 6]
            quaternion_update = F.pad(
                quaternion_update,
                (1, 0),
                value=1.,
            ) # pad只有两个参数时, 代表对最后一个维度进行改变, 这句话意思是对左边扩充1列，右边扩充0列,填充值是1

            quaternions = quaternion_multiply(
                quaternions,
                quaternion_update,
            )  # 利用quaternion_update以及四元数矩阵乘法，对quaternions旋转矩阵进行更新
            translations = translations + einsum(
                'b n c, b n c r -> b n r',  # 这里的c和r都是3
                translation_update,
                rotations,
            )  # 将局部frame下的偏移向量更新（translation_update）与全局frame旋转矩阵相乘，得到全局frame下的偏移向量

        # 这里得到的是4个骨架原子集合ideal_coords，并且将这种骨架原子进行repeat，以实现对每个氨基酸骨架原子的坐标初始化（在更改模型时，这里不需要更改）
        ideal_coords = get_ideal_coords().to(device)
        ideal_coords = repeat(
            ideal_coords,
            "a d -> b l a d",
            b=b,
            l=n,
        )  # [b, r, 4, 3]  a=4, 代表4种骨架重原子, d=3 是因为是三维坐标

        rotations = quaternion_to_matrix(quaternions)  # [b, r, 3, 3]
        points_global = einsum(
            'b n a c, b n c d -> b n a d',
            ideal_coords,
            rotations,
        ) + rearrange(
            translations,
            "b l d -> b l () d",
        )  # 利用全局frame下的Ti得到每个氨基酸的坐标

        points_global = place_o_coords(points_global)

        if not self.require_nodes_feat:
            return points_global, translations, quaternions
        else:
            return points_global, translations, quaternions, x
