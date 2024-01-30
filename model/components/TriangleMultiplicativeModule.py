###
#   Inspired by triangle multiplicative update implementation from https://github.com/lucidrains/triangle-multiplicative-module
###

from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

from utils.general import exists, default


class TriangleMultiplicativeModule(nn.Module):
    def __init__(
        self,
        *,
        dim,
        hidden_dim=None,
        mix='ingoing',
    ):
        super().__init__()
        assert mix in {'ingoing', 'outgoing'}, 'mix must be either ingoing or outgoing'

        hidden_dim = default(
            hidden_dim,
            dim,
        )
        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim,)
        self.right_proj = nn.Linear(dim, hidden_dim,)

        self.left_gate = nn.Linear(dim, hidden_dim,)
        self.right_gate = nn.Linear(dim, hidden_dim,)
        self.out_gate = nn.Linear(dim, dim)

        # initialize all gating to be identity

        for gate in (
                self.left_gate,
                self.right_gate,
                self.out_gate,
        ):
            nn.init.constant_(gate.weight, 0.,)  # 对于Gated层，需要将参数矩阵初始化为0，bias初始化为1
            nn.init.constant_(gate.bias, 1.,)

        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'ingoing':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim,)

    def forward(self, x, mask=None):
        assert x.shape[1] == x.shape[2], 'feature map must be symmetrical'  # x:[b, r, r, cz]
        if exists(mask):  # mask: [b, r, r]
            mask = rearrange(
                mask,
                'b i j -> b i j ()',
            )  # mask: [b, r, r, 1]

        x = self.norm(x)  # 对pair_rep中的每个z_ij进行affine

        left = self.left_proj(x)
        right = self.right_proj(x)

        if exists(mask):
            left = left * mask  # [b, r, r, h_cz]
            right = right * mask  # [b, r, r, h_cz]

        left_gate = self.left_gate(x).sigmoid()  # 利用Linear以及sigmoid，产生了3个gate
        right_gate = self.right_gate(x).sigmoid()  # tensor： [b, r, r, h_cz]
        out_gate = self.out_gate(x).sigmoid()  # [b, r, r, cz]

        left = left * left_gate  # [b, r, r, h_cz]
        right = right * right_gate  # [b, r, r, h_cz]

        out = einsum(  # out:[b, r, r, h_cz]
            self.mix_einsum_eq,
            left,
            right,
        )  # 本质是对edge也就是pair_rep进行更新，更新方法是收集所有以i,j作为出点(或入点)并同时指向k这一个点的两个三角边的相乘信息的sum作为当前e_ij更新值

        out = self.to_out_norm(out)  # 进行LayerNorm
        out = self.to_out(out)  # 利用Linear，恢复为cz [b, r, r, cz]
        out = out * out_gate  # [b, r, r, cz]
        return out
