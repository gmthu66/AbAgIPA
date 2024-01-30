###
#   Inspired by graph transformer implementation from https://github.com/lucidrains/graph-transformer-pytorch
###

import torch
from torch import nn, einsum
from einops import rearrange, repeat

from utils.general import exists, default

List = nn.ModuleList


class PreNorm(nn.Module):

    def __init__(
        self,
        dim,
        fn,
    ):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x,
        *args,
        **kwargs,
    ):
        x = self.norm(x)
        return self.fn(
            x,
            *args,
            **kwargs,
        )


# gated residual


class Residual(nn.Module):

    def forward(
        self,
        x,
        res,
    ):
        return x + res


class GatedResidual(nn.Module):

    def __init__(
        self,
        dim,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, res):
        gate_input = torch.cat((x, res, x - res), dim=-1)  # 这里的gate来自于x, 上一层的x也就是res以及x-res, 因此self.proj中的Linear层输入channel为dim*3
        gate = self.proj(gate_input)
        return x*gate + res * (1-gate)


# attention


class Attention(nn.Module):
    # 输入是nodes以及edges，这里是（在k和v中）引入了pair_wise_rep, q仍然为[b, r, cm]， 但是k和v都分别与ek和ev进行了相加[b, r, r, cm]
    # 实现的是将蛋白质的氨基酸特征看作一个全连接图，利用氨基酸i的节点特征m_i与i为出节点的所有edge特征e_ij进行内积得到节点向量，作为attn_score的a_ij，最终得到[b, r, r]大小的a
    def __init__(
        self,
        dim,
        dim_head=64,  # 每个head的dim
        heads=8,
        edge_dim=None,
    ):
        super().__init__()
        edge_dim = default(
            edge_dim,
            dim,
        )

        inner_dim = dim_head * heads  # (32*8) inner_dim = 每个注意力head的dim再乘以heads数目, 此处为256
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(
            dim,
            inner_dim,
        )
        self.to_kv = nn.Linear(
            dim,
            inner_dim * 2,
        )
        self.edges_to_kv = nn.Linear(
            edge_dim,
            inner_dim,
        )

        self.to_out = nn.Linear(
            inner_dim,
            dim,
        )

    def forward(
        self,
        nodes,
        edges,
        mask=None,  # nodes:[b, r, cm]  edges:[b, r, r, cz], cm=cz
    ):  # 利用初始化的edge信息计算nodes之间的注意力
        h = self.heads

        q = self.to_q(nodes) # q: [b, r, (c * h)]， h为head数目
        # k和v: [b, r, (c * h)]， h为head数目
        k, v = self.to_kv(nodes).chunk(2, dim=-1,)  #.chunk是对tensor进行分块，返回一个张量列表

        e_kv = self.edges_to_kv(edges)  # [b, r, r, (c * h)] 对pair_feat进行affine, c就是下面的特征维度d

        q, k, v, e_kv = map(
            lambda t: rearrange(
                t,
                'b ... (h d) -> (b h) ... d',
                h=h,
            ),
            (q, k, v, e_kv),
        )  # q,k,v: [b*h, r, c]     e_kv:[b*h, r, r, c]

        ek, ev = e_kv, e_kv  # pair_rep得到的key和value直接采用同一个tensor
        # 将edge_pair特征分别注入到key和value中
        k, v = map(
            lambda t: rearrange(
                t,
                'b j d -> b () j d ',
            ),
            (k, v),
        )  # 利用rearrange进行unsqueeze操作，在氨基酸索引位置之前再添加一个空白维度
        k = k + ek  # 将edge_pair特征分别注入到key和value中，进行了expand， k： [b*h, r, r, c]
        v = v + ev  # v: [b*h, r, r, c]

        sim = einsum(
            'b i d, b i j d -> b i j',
            q,
            k,
        ) * self.scale  # 利用einusm实现attn_score计算, 得到sim：[b*h, r, r]

        # 进行注意力计算
        if exists(mask):
            mask = rearrange(
                mask,
                'b i -> b i ()',
            ) & rearrange(
                mask,
                'b j -> b () j',
            )  # 从序列的mask，类似于进行了叉乘，得到了二维的横纵分别都是氨基酸索引的新mask
            mask = repeat(
                mask,
                "b ... -> (b h) ...",
                h=self.heads,
            )
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)  # 为attention计算过程中被mask掉的ij位置进行负的最大值赋值, 使其不参与注意力分数的计算
            # sim: (b*h, r, r)
        attn = sim.softmax(dim=-1)  # [b*h, r, r] 对于node_i不变, 与node_i之间有关系的e_ij部分(j是取了其他所有位置的氨基酸)都拿过来进行了注意力计算
        out = einsum(
            'b i j, b i j d -> b i d',
            attn,
            v,
        )  # [b*h, r, r]  对于固定的i， 利用attn_ij和value_ij，针对j遍历进行weighted_sum得到更新后的nodes：[b*h, i, d]
        out = rearrange(
            out,
            '(b h) n d -> b n (h d)',
            h=h,
        )  # 恢复tensor变为 [b, r, h*d]，  其中h*d=cm
        return self.to_out(out)


def FeedForward(dim, ff_mult=4):  # ff_mul代表FFN中隐含层的hid_dim针对输入dim的倍数
    return nn.Sequential(
        nn.Linear(dim, dim * ff_mult),
        nn.GELU(),
        nn.Linear(dim * ff_mult, dim),
    )


class GraphTransformer(nn.Module):
    # 输入是nodes以及edges
    def __init__(
        self,
        dim,
        depth,
        dim_head=64,
        edge_dim=None,
        heads=8,
        with_feedforwards=False,
        norm_edges=False,
    ):
        super().__init__()
        self.layers = List([])  # 这里的List都是nn.ModuleList
        edge_dim = default(
            edge_dim,
            dim,
        )
        self.norm_edges = nn.LayerNorm(edge_dim) if norm_edges else nn.Identity()  # nn.Identity()是一个用来占位的网络层，不起到实际效果

        for _ in range(depth):
            self.layers.append(
                List([
                    List([
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                edge_dim=edge_dim,
                                dim_head=dim_head,
                                heads=heads,
                            )
                        ),  # PreNorm是自定义的, 将LayerNorm和后续一个网络模块相联系的Module。参数为PreNorm(dim, fn)，fn指的是需要在norm后执行的网络层
                        GatedResidual(dim)
                    ]),
                    List([
                        PreNorm(
                            dim,
                            FeedForward(dim),
                        ),
                        GatedResidual(dim)
                    ]) if with_feedforwards else None
                ])
            )

    def forward(
        self,
        nodes,
        edges,
        mask=None,
    ):
        edges = self.norm_edges(edges)  # 占位，其实没有做norm

        for attn_block, ff_block in self.layers:
            # attn_block是以PreNorm(Atten())和Gated残差连接为核心的注意力计算块， ff_block是以PreNorm(FFN)和Gated残差连接为核心的前向传播块
            attn, attn_residual = attn_block
            nodes = attn_residual(
                attn(
                    nodes,
                    edges,
                    mask=mask,
                ),
                nodes,
            )

            if exists(ff_block):
                ff, ff_residual = ff_block
                nodes = ff_residual(
                    ff(nodes),
                    nodes,
                )

        return nodes, edges  # 返回的edges并没有被使用
