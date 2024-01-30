import torch
from torch import nn

from .GraphTransformer import GraphTransformer
from .TriangleMultiplicativeModule import TriangleMultiplicativeModule
from utils.general import exists


class TriangleGraphTransformer(nn.Module):
    def __init__(
        self,
        dim,
        edge_dim,
        depth,
        gt_depth=1,
        gt_dim_head=32,
        gt_heads=8,
        tri_dim_hidden=None,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            graph_transformer = GraphTransformer(
                dim=dim,
                edge_dim=edge_dim,
                depth=gt_depth,
                heads=gt_heads,
                dim_head=gt_dim_head,
                with_feedforwards=True,
            )
            # triangle_out以及triangle_in是对edges_ij进行更新的
            triangle_out = TriangleMultiplicativeModule(
                dim=edge_dim,
                hidden_dim=tri_dim_hidden,
                mix='outgoing',
            )
            triangle_in = TriangleMultiplicativeModule(
                dim=edge_dim,
                hidden_dim=tri_dim_hidden,
                mix='ingoing',
            )

            self.layers.append(
                nn.ModuleList([graph_transformer, triangle_out, triangle_in]))

    def forward(self, nodes, edges, mask=None):  # nodes:[b, l, cm]  edges:[b, l, l, cz]  mask:[b, l]
        for gt, tri_out, tri_in in self.layers:  # gt:GraphTransformer  tri_out:TriangleMultiplicativeModule
            if exists(mask):
                tri_mask = mask.unsqueeze(-2) & mask.unsqueeze(-1)
            else:
                tri_mask = None

            nodes, _ = gt(nodes, edges, mask=mask)  # edges作为邻接矩阵用于nodes的特征更新(三角形乘法更新)
            edges = edges + tri_out(
                edges,
                mask=tri_mask,
            )
            edges = edges + tri_in(
                edges,
                mask=tri_mask,
            )

        return nodes, edges
