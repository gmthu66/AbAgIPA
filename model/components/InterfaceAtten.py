import torch
from torch import nn, einsum
from einops import rearrange, repeat

from utils.general import exists, default

List = nn.ModuleList


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs,):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class Residual(nn.Module):
    def forward(self, x, res):
        return x + res


class GatedResudual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim * 3, 1, bias=False),
                                  nn.Sigmoid(),)


class InterAtten(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, edge_dim=None):
        super().__init__()
        edge_dim = default(edge_dim, dim)

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, ab_nodes, at_nodes, ab_mask=None, at_mask=None):
        h = self.heads
        q = self.to_q(ab_nodes)
        k = self.to_k(at_nodes)
        v = self.to_v(ab_nodes)

        q, k, v = map(lambda t: rearrange(t, 'b, ... (h, d) -> (b h) ... d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        if exists(mask):
            mask = rearrange(ab_mask, 'b i -> b i ()') & rearrange(at_mask, 'b j -> b j ()')
            mask = repeat(mask, 'b ... -> (b h) ...')
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)  # 为attention计算过程中被mask掉的ij位置进行负的最大值赋值, 使其不参与注意力分数的计算
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b i d')


class InterEdgeAtten(nn.Module):
    def __init__(self, edge_dim, dim_head=64, heads=4):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(edge_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(edge_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(edge_dim, inner_dim, bias=False)
        self.to_ebias = nn.Linear(edge_dim, 1 * heads, bias=False)
        self.to_gate = nn.Sequential(nn.Linear(edge_dim, inner_dim),
                                     nn.Sigmoid())

        self.to_out = nn.Linear(inner_dim, edge_dim)

    def forward(self, inter_edges, ab_mask=None, at_mask=None, mask=None):
        # torch.cat([str_nodes, at_str_nodes], dim=-1)
        # inter_edges = torch.cat(str_nodes, at_str_nodes)
        h = self.heads
        q = self.to_q(inter_edges)
        k = self.to_k(inter_edges)
        v = self.to_v(inter_edges)
        q, k, v = map(lambda t: rearrange(t, 'b ... (h d) -> (b h) ... d', h=h), (q, k, v))
        z_bias = self.to_ebias(inter_edges).transpose(0, -1)
        z_gate = self.to_gate(inter_edges)
        z_gate = rearrange(z_gate, 'b i j (d h) -> (b h) i j d', h=self.heads)

        sim = einsum('h i j d, h i k d -> h i j k', q, k) * self.scale
        sim = sim + z_bias
        if mask is None:
            mask = rearrange(ab_mask, 'b i -> b i ()')  & rearrange(at_mask, 'b j -> b () j')
            mask = repeat(mask, 'b ... -> (b h) ...', h=self.heads)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask.unsqueeze(-1), max_neg_value)
        attn = sim.softmax(dim=-1)

        out = einsum('h i j k, h i j d -> h i j d', attn, v)
        out = einsum('h i j d, h i j d -> h i j d', z_gate, out)
        out = self.to_out(rearrange(out, '(b h) i j d -> b i j (h d)', h=self.heads))
        return out


class InterEdge(nn.Module):
    def __init__(self, depth, edge_dim=None, dim_head=64, heads=4):
        super().__init__()
        self.Layers = List([])
        edge_dim = default(edge_dim, dim_head,)

        for _ in range(depth):
            self.Layers.append(List([
                PreNorm(
                    edge_dim, 
                    InterEdgeAtten(
                        edge_dim,
                        dim_head=dim_head,
                        heads=heads)),
                        Residual()]))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, inter_edges, ab_mask=None, at_mask=None, mask=None):
        for attn_block, attn_residual in self.Layers:
            inter_edges = attn_residual(inter_edges, attn_block(inter_edges, ab_mask=ab_mask, at_mask=at_mask))
        inter_edges = self.pool(inter_edges.permute(0, 3, 1, 2).contiguous())
        return inter_edges


class MaskMaxPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat, mask):
        # feat = feat.masked_fill(~mask.unsqueeze(-1), 0.0)
        feat = feat * mask.unsqueeze(-1)
        feat, _ = torch.max(feat, dim=1)
        return feat


class MaskMeanPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat, mask):
        # feat = feat.masked_fill(~mask.unsqueeze(-1), 0.0)
        feat = feat * mask.unsqueeze(-1)
        nodes_num = mask.sum(dim=1, keepdim=True)
        feat = feat.sum(dim=1) / nodes_num
        return feat


class Inter_OnlyAbGlobalMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool1 = MaskMeanPool()
        self.pool2 = MaskMeanPool()

    def forward(self, ab_nodes, ab_mask=None):
        #  ab_nodes
        ab_nodes = self.pool1(ab_nodes, ab_mask)
        # inter_feat = ab_nodes * at_nodes
        return ab_nodes


class Inter_GlobalMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool1 = MaskMeanPool()
        self.pool2 = MaskMeanPool()

    def forward(self, ab_nodes, at_nodes, ab_mask=None, at_mask=None):
        #  ab_nodes
        ab_nodes = self.pool1(ab_nodes, ab_mask)
        at_nodes = self.pool2(at_nodes, at_mask)
        # inter_feat = ab_nodes * at_nodes
        inter_feat = torch.cat([ab_nodes, at_nodes], dim=1)
        return inter_feat


class Inter_GlobalMax(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool1 = MaskMaxPool()
        self.pool2 = MaskMaxPool()

    def forward(self, ab_nodes, at_nodes, ab_mask=None, at_mask=None):
        #  ab_nodes
        ab_nodes = self.pool1(ab_nodes, ab_mask)
        at_nodes = self.pool2(at_nodes, at_mask)
        # inter_feat = ab_nodes * at_nodes
        inter_feat = torch.cat([ab_nodes, at_nodes], dim=1)
        return inter_feat


class Inter_1dCNN(nn.Module):
    def __init__(self, in_dim, hid_dim, kernel_size=1, numlayers=2, num_points=1, norm_input=False, rot_dim=4, point_bias=True, num_classes=1, norm='none', dropout=False):
        super().__init__()
        self.layers = nn.ModuleList()
        if norm_input:
            self.layers.append(nn.BatchNorm1d(in_dim))
        for i_ in range(numlayers):
            _in_dim = in_dim if i_ == 0 else hid_dim
            self.layers.append(nn.Conv1d(_in_dim, hid_dim, kernel_size))
            self.layers.append(nn.BatchNorm1d(hid_dim))
            self.layers.append(nn.ReLU())
            if dropout:
                self.layers.append(nn.Dropout(p=0.2))
        self.neck = nn.ModuleList()
        self.neck.append(nn.Conv1d(hid_dim, rot_dim * num_classes, 1))
        self.conv_p = nn.Conv1d(num_points, 1, 1, bias=point_bias)

    def forward(self, x):
        for _layer in self.layers:
            x = _layer(x)

        for _layer in self.neck:
            x = _layer(x)

        feat = x.clone()
        x = x.permute(0, 2, 1)
        x = self.conv_p(x)

        x = x.squeeze(1)
        x = x.contiguous()

        return x, feat


class InterAttenBlock(nn.Module):
    def __init__(self, dim, hdim):
        super().__init__()
        self.to_q = nn.Linear(dim, hdim)
        self.to_k = nn.Linear(dim, hdim)
        self.to_v = nn.Linear(dim, hdim)
        # self.scalar_attn = hdim ** -0.5

        self.mean_pool = MaskMeanPool()

    def forward(self, ab_nodes, at_nodes, ab_mask=None, at_mask=None):
        mean_at = self.mean_pool(at_nodes, at_mask)
        q = self.to_q(ab_nodes)
        k = self.to_k(mean_at)
        v = self.to_v(ab_nodes)

        attn_logits = einsum('b i d, j d -> b i', q, k)
        attn_logits = attn_logits.masked_fill(~ab_mask, -1e9)
        # attn = attn_logits.softmax(dim = - 1) * self.scalar_attn
        attn = attn_logits.softmax(dim = - 1)
        v = einsum('b i d, b i -> b d', v, attn)
        return v, attn


class Inter_SoftAtten(nn.Module):
    def __init__(self, dim, hdim, head=1, depth=1):
        super().__init__()
        self.nlayer = depth
        self.attn_norms = nn.ModuleList([nn.LayerNorm(hdim) for _ in range(depth)])
        self.layers = nn.ModuleList([InterAttenBlock(dim, hdim) for _ in range(depth)])
        self.inter_globalMean = Inter_GlobalMean()

    def forward(self, ab_nodes, at_nodes, ab_mask=None, at_mask=None):
        for i_, attnlayer in enumerate(self.layers):
            ab_nodes_res, attn = self.layers[0](ab_nodes, at_nodes, ab_mask, at_mask)
            ab_nodes = ab_nodes_res.unsqueeze(1) + ab_nodes
            ab_nodes = self.attn_norms[i_](ab_nodes)

        # at_nodes = self.mean_pool(at_nodes, at_mask)
        # inter_feat = ab_nodes * at_nodes
        # at_nodes = self.mean_pool(at_nodes, at_mask)
        inter_feat = self.inter_globalMean(ab_nodes, at_nodes, ab_mask, at_mask)
        return inter_feat


class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.5):
        super(BinaryClassifier, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.bn1 = nn.BatchNorm1d(hidden_size)  # 可选的批归一化层
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 2)  # 输出为1，因为是二分类任务

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)  # 如果使用批归一化，则在激活函数之前应用
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        # return torch.sigmoid(x)  # 二分类任务通常使用sigmoid激活函数
        return x  # 二分类任务通常使用sigmoid激活函数


def get_nn_act_func(act, inplace=True, **kwargs):
    """Using torch.nn if possible."""
    if act is None:
        return nn.Identity()

    if act.lower() == "relu":
        act_func = nn.ReLU(inplace=inplace)
    elif act.lower() == "sigmoid":
        act_func = nn.Sigmoid()
    elif act.lower() == "prelu":
        # num_parameters=1, init=0.25
        act_func = nn.PReLU(**kwargs)
    elif act.lower() in ["lrelu", "leaky_relu", "leakyrelu"]:
        kwargs.setdefault("negative_slope", 0.1)
        act_func = nn.LeakyReLU(inplace=inplace, **kwargs)
    elif act.lower() in ["silu", "swish"]:
        # requires pytorch>=1.7
        act_func = nn.SiLU(inplace=inplace)
    elif act.lower() == "gelu":
        act_func = nn.GELU()
    elif act.lower() == "mish":
        act_func = nn.Mish(inplace=inplace)
    elif len(act) == 0 or act.lower() == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation function: {act}.")
    return act_func


# def get_norm(norm, out_channels, num_gn_groups=32):
#     """
#     Args:
#         norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
#             or a callable that takes a channel number and returns
#             the normalization layer as a nn.Module.
#     Returns:
#         nn.Module or nn.Identity(): the normalization layer
#     """
#     if norm is None:
#         return nn.Identity()
#     if isinstance(norm, str):
#         if len(norm) == 0 or norm.lower() == "none":
#             return nn.Identity()
#         norm = {
#             "BN": BatchNorm2d,
#             "BN1d": nn.BatchNorm1d,
#             # Fixed in https://github.com/pytorch/pytorch/pull/36382
#             "SyncBN": NaiveSyncBatchNorm if env.TORCH_VERSION <= (1, 5) else nn.SyncBatchNorm,
#             "FrozenBN": FrozenBatchNorm2d,
#             "GN": lambda channels: nn.GroupNorm(num_gn_groups, channels),
#             "IN": nn.InstanceNorm2d,
#             # for debugging:
#             "nnSyncBN": nn.SyncBatchNorm,
#             "naiveSyncBN": NaiveSyncBatchNorm,
#         }[norm]
#     return norm(out_channels)