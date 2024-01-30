###
#   Inspired by IPA implementation from https://github.com/lucidrains/invariant-point-attention
###

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from contextlib import contextmanager
from torch import nn, einsum

from einops.layers.torch import Rearrange
from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

@contextmanager
def disable_tf32():
    orig_value = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = orig_value

# classes

class InvariantPointAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        scalar_key_dim = 16,
        scalar_value_dim = 16,
        point_key_dim = 4,
        point_value_dim = 4,
        pairwise_repr_dim = None,
        require_pairwise_repr = True,
        eps = 1e-8,
        with_point_attn = False,
        return_attn = False,
    ):
        super().__init__()
        self.eps = eps
        self.heads = heads
        self.require_pairwise_repr = require_pairwise_repr
        self.with_point_attn = with_point_attn
        self.return_attn = return_attn

        # num attention contributions

        num_attn_logits = 3 if require_pairwise_repr else 2  # 计算attn时，所参与输入类型数目

        # qkv projection for scalar attention (normal)

        self.scalar_attn_logits_scale = (num_attn_logits * scalar_key_dim) ** -0.5  # 用来产生scalar_value对应的 1/sqrt(d_k)

        self.to_scalar_q = nn.Linear(dim, scalar_key_dim * heads, bias = False)
        self.to_scalar_k = nn.Linear(dim, scalar_key_dim * heads, bias = False)
        self.to_scalar_v = nn.Linear(dim, scalar_value_dim * heads, bias = False)

        # qkv projection for point attention (coordinate and orientation aware)

        point_weight_init_value = torch.log(torch.exp(torch.full((heads,), 1.)) - 1.)
        self.point_weights = nn.Parameter(point_weight_init_value)  # 这里是对point_weight进行初始设置

        self.point_attn_logits_scale = ((num_attn_logits * point_key_dim) * (9 / 2)) ** -0.5

        self.to_point_q = nn.Linear(dim, point_key_dim * heads * 3, bias = False)  # 最后得到的是三维坐标, point_key_dim默认是4代表每个氨基酸上的4个骨架原子
        self.to_point_k = nn.Linear(dim, point_key_dim * heads * 3, bias = False)
        self.to_point_v = nn.Linear(dim, point_value_dim * heads * 3, bias = False)  # 3代表三维坐标

        # pairwise representation projection to attention bias

        pairwise_repr_dim = default(pairwise_repr_dim, dim) if require_pairwise_repr else 0

        if require_pairwise_repr:
            self.pairwise_attn_logits_scale = num_attn_logits ** -0.5

            self.to_pairwise_attn_bias = nn.Sequential(
                nn.Linear(pairwise_repr_dim, heads),
                Rearrange('b ... h -> (b h) ...')
            )  # 将pair_rep affine到一个数值作为attn_score的偏置量，对于每个head都有一个偏置量

        # combine out - scalar dim + pairwise dim + point dim * (3 for coordinates in R3 and then 1 for norm)
        # self.to_out = nn.Linear(heads * (scalar_value_dim + pairwise_repr_dim + point_value_dim * (3 + 1)), dim)
        if not self.with_point_attn:
            self.to_out = nn.Linear(heads * (scalar_value_dim + pairwise_repr_dim), dim)
        else:
            self.to_out = nn.Linear(heads * (scalar_value_dim + pairwise_repr_dim + point_value_dim * (3 + 1)), dim)

    def forward(  # 输入主要是目标氨基酸特征矩阵, pair_rep以及每个氨基酸的旋转矩阵（四元数形式）与偏移向量Ti=(Ri, ti)
        self,
        single_repr,  # [b, r, cm]
        pairwise_repr = None,  # [b, r, r, cz]
        *,
        rotations,  # [b, r, 3, 3] 是根据template得到的rotations旋转矩阵，被看做属于global frame下的变换
        translations,  # [b, r, 3] 是根据template得到的偏移向量矩阵，被看做属于global frame下的变换
        mask = None
    ):
        x, b, h, eps, require_pairwise_repr = single_repr, single_repr.shape[0], self.heads, self.eps, self.require_pairwise_repr
        assert not (require_pairwise_repr and not exists(pairwise_repr)), 'pairwise representation must be given as second argument'

        # get queries, keys, values for scalar and point (coordinate-aware) attention pathways

        q_scalar, k_scalar, v_scalar = self.to_scalar_q(x), self.to_scalar_k(x), self.to_scalar_v(x)  # [b, r, c_scalar * h] 其中 h=8, 来自于 single_feat(目标氨基酸序列特征矩阵)
        # q_point等变量代表每个氨基酸上各个4个重原子的三维坐标，由 single_repr 经过affine得到
        q_point, k_point, v_point = self.to_point_q(x), self.to_point_k(x), self.to_point_v(x)  # [b, r, c_point * h * 3] 其中h=8, 3代表三维坐标维度

        # split out heads 切分tensor得到head维度

        q_scalar, k_scalar, v_scalar = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q_scalar, k_scalar, v_scalar))  # [b*h, r, c_scalar]
        q_point, k_point, v_point = map(lambda t: rearrange(t, 'b n (h d c) -> (b h) n d c', h = h, c = 3), (q_point, k_point, v_point))  # [b*h, r, c_point, 3], d代表每个氨基酸的4个重原子
        # rotations旋转矩阵和translations偏移向量来自于模板结构或者来自于预测网络中上一层输出的不断迭代更新的Ti=(Ri, ti)
        rotations = repeat(rotations, 'b n r1 r2 -> (b h) n r1 r2', h = h)  # 将旋转矩阵repeat到不同的head中： [b*h, r, 3, 3]
        translations = repeat(translations, 'b n c -> (b h) n () c', h = h)  # 将平移向量repeat到不同的head中： [b*h, r, 1, 3]

        # rotate qkv points into global frame, 本段的b其实是b*h

        q_point = einsum('b n d c, b n c r -> b n d r', q_point, rotations) + translations # einsum操作是基于rotations对本模块网络得到的q_point坐标进行旋转,然后加上平移向量
        k_point = einsum('b n d c, b n c r -> b n d r', k_point, rotations) + translations
        v_point = einsum('b n d c, b n c r -> b n d r', v_point, rotations) + translations

        # derive attn logits for scalar and pairwise

        attn_logits_scalar = einsum('b i d, b j d -> b i j', q_scalar, k_scalar) * self.scalar_attn_logits_scale  # einsum实现的是dot_product（即氨基酸i和氨基酸j的特征向量的内积作为attn_ij的score）

        if require_pairwise_repr:  # 是否需要基于pair_rep生成对于旋转矩阵的偏置量
            attn_logits_pairwise = self.to_pairwise_attn_bias(pairwise_repr) * self.pairwise_attn_logits_scale  # 从[b, r, r, cz]这种pairrep转化为[b*h, r, r]这种sacaler用于后续加和

        # derive attn logits for point attention

        point_qk_diff = rearrange(q_point, 'b i d c -> b i () d c') - rearrange(k_point, 'b j d c -> b () j d c')  # 从q_point和k_point得到两者的坐标向量diff
        point_dist = (point_qk_diff ** 2).sum(dim = -2)  # 得到p_point与k_point的四种重原子距离总的差值, 将[b, r, r, 4, 3]转化为[b, r, r, 3]

        point_weights = F.softplus(self.point_weights)  # softplus是ReLU函数的平滑近似，可以使得输出约束始终为正(对于每个head的注意力weight都是正的)
        point_weights = repeat(point_weights, 'h -> (b h) () () ()', b = b)

        attn_logits_points = -0.5 * (point_dist * point_weights * self.point_attn_logits_scale).sum(dim = -1)  # 这里更侧重于局部的四个重原子坐标是否准确
        # 乘以负值是因为point_dist越大，两个氨基酸上的重原子之间的注意力分数理应越小
        # combine attn logits

        attn_logits = attn_logits_scalar + attn_logits_points

        if require_pairwise_repr:
            attn_logits = attn_logits + attn_logits_pairwise

        # mask

        if exists(mask):
            mask = rearrange(mask, 'b i -> b i ()') * rearrange(mask, 'b j -> b () j')
            mask = repeat(mask, 'b i j -> (b h) i j', h = h)  # 得到[b*h, r, r]大小的mask_tensor，即i和j中任一结点被mask掉时,ij对应的attn_logit得分也被mask掉
            mask_value = max_neg_value(attn_logits)
            attn_logits = attn_logits.masked_fill(~mask, mask_value)
            if self.return_attn:
                attn_logits_scalar = attn_logits_scalar.masked_fill(~mask, mask_value)
                attn_logits_scalar = attn_logits_scalar.softmax(dim = - 1)
                attn_logits_points = attn_logits_points.masked_fill(~mask, mask_value)
                attn_logits_points = attn_logits_points.softmax(dim = - 1)
                attn_scalar_withhead = rearrange(attn_logits_scalar, '(b h) i j -> b h i j', h = h)
                attn_points_withhead = rearrange(attn_logits_points, '(b h) i j -> b h i j', h = h)
                attn_scalar = attn_scalar_withhead.mean(1)
                attn_points = attn_points_withhead.mean(1)
        # attention

        attn = attn_logits.softmax(dim = - 1)

        with disable_tf32(), autocast(enabled = False):
            # disable TF32 for precision （将torch.backends.cuda.matmul.allow_tf32设置为False,提高计算精度，为True时是会提高计算效率） #autocast(enabled=False)代表将自动半精度加速训练关闭 

            # aggregate values(利用融合pair_rep的AttnBias、nodefeat的AttnScalar、融合旋转与平移矩阵的nodefeate产生的局部point对应的attn_point)
            # 利用当前i与每个j之间的attn得分([h, i, j]) 融合其他结点的sg_feat特征[h, j, c]到i上，得到更新后的特征矩阵[h, r, c]， 操作就是正常的矩阵相乘
            results_scalar = einsum('b i j, b j d -> b i d', attn, v_scalar)  # 这里的b是b*h

            attn_with_heads = rearrange(attn, '(b h) i j -> b h i j', h = h)

            if require_pairwise_repr:  # 若需要来自pair_rep的attn_bias信息, 则同样地，基于attn_logits[b, h, i, j]对[b, r, r, cz]进行特征融合得到 [b, r, h, c]
                results_pairwise = einsum('b h i j, b i j d -> b h i d', attn_with_heads, pairwise_repr)

            # aggregate point values # attn_ij是针对i和j氨基酸size的,针对氨基酸i，利用atten_ij对各个j的重原子坐标进行加权求和来更新ponit_ij,得到[r, h, p, 3]
            if self.with_point_attn:
                results_points = einsum('b i j, b j d c -> b i d c', attn, v_point)  # 这里的b其实是 b*h, 操作其实是[b, i, j]对于[b, j, d*c]进行矩阵相乘然后再恢复d和c的维度位置

                # rotate aggregated point values back into local frame # 将得到的point坐标再减去translations，并乘以global的旋转矩阵的转置恢复为 local frame
                results_points = einsum('b n d c, b n c r -> b n d r', results_points - translations, rotations.transpose(-1, -2))
                results_points_norm = torch.sqrt(torch.square(results_points).sum(dim=-1) + eps )  # torch.square(results_points).sum(dim=-1)是将重原子向量各个分量平方求和再开方

        # merge back heads

        results_scalar = rearrange(results_scalar, '(b h) n d -> b n (h d)', h = h)  # 进行各个维度的恢复
        if self.with_point_attn:
            results_points = rearrange(results_points, '(b h) n d c -> b n (h d c)', h = h)  # h=8, d=4, c=3，3是三维坐标, 4是4个重原子  (将坐标信息删掉了, 可能过于冗余了)
            results_points_norm = rearrange(results_points_norm, '(b h) n d -> b n (h d)', h = h)
            results = (results_scalar, results_points, results_points_norm)

        if require_pairwise_repr:
            results_pairwise = rearrange(results_pairwise, 'b h n d -> b n (h d)', h = h)
            if self.with_point_attn:
                results = (*results, results_pairwise)
                results = torch.cat(results, dim = -1)
            else:
                results = torch.cat([results_scalar, results_pairwise], dim = -1)  # 在这里是把
        else:
            if not self.with_point_attn:
                results = results_scalar
            else:
                results = torch.cat(results, dim = -1)
        # concat results and project out  将 results_scalar, results_points, results_points_norm 在氨基酸特征维度上进行concate
        # results = torch.cat(results, dim = -1)  # 在这里是把
        if not self.return_attn:
            return self.to_out(results)
        else:
            attn_logits_scalar = attn_scalar_withhead.softmax(dim = - 1)

            attn_with_heads.mean(dim=1)
            return self.to_out(results), attn_scalar, attn_points

# one transformer block based on IPA

def FeedForward(dim, mult = 1., num_layers = 2, act = nn.ReLU):
    layers = []
    dim_hidden = dim * mult

    for ind in range(num_layers):
        is_first = ind == 0
        is_last  = ind == (num_layers - 1)
        dim_in   = dim if is_first else dim_hidden
        dim_out  = dim if is_last else dim_hidden

        layers.append(nn.Linear(dim_in, dim_out))

        if is_last:
            continue

        layers.append(act())

    return nn.Sequential(*layers)

class IPABlock_ForInter(nn.Module):
    def __init__(
        self,
        *,
        dim,
        ff_mult = 1,
        ff_num_layers = 3,     # in the paper, they used 3 layer transition (feedforward) block
        post_norm = True,      # in the paper, they used post-layernorm - offering pre-norm as well
        return_attn = False,
        **kwargs
    ):
        super().__init__()
        self.post_norm = post_norm
        self.return_attn = return_attn

        self.attn_norm = nn.LayerNorm(dim)
        self.attn = InvariantPointAttention(dim = dim, return_attn=return_attn, **kwargs)

        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult = ff_mult, num_layers = ff_num_layers)

    def forward(self, x, **kwargs):
        post_norm = self.post_norm

        attn_input = x if post_norm else self.attn_norm(x)  # 如果是先算attn再norm，则attn_input在此处先不做norm操作
        if not self.return_attn:
            x = self.attn(attn_input, **kwargs) + x  # 利用点不变注意力机制更新single_feat，IPA的输入是：single_repr，pairwise_repr，rotations和translations
        else:
            tempx, res_attn, point_attn = self.attn(attn_input, **kwargs)  # 利用点不变注意力机制更新single_feat，IPA的输入是：single_repr，pairwise_repr，rotations和translations
            x = tempx + x
        x = self.attn_norm(x) if post_norm else x  # 若post_norm为真, 则在此处做norm操作, norm_dim为64

        ff_input = x if post_norm else self.ff_norm(x)
        x = self.ff(ff_input) + x
        x = self.ff_norm(x) if post_norm else x
        if not self.return_attn:
            return x
        else:
            return x, res_attn, point_attn
