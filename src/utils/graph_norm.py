## Adapted from: https://github.com/cyh1112/GraphNormalization

import torch.nn as nn
import torch


class GraphNorm(nn.Module):
    """
        Param: []
    """
    def __init__(self, num_features, eps=1e-5, affine=True, is_node=True):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.affine = affine
        self.is_node = is_node

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.num_features))
            self.beta = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def norm(self, x):
        mean = x.mean(dim = 0, keepdim = True)
        var = x.std(dim = 0, keepdim = True)
        return (x - mean) / (var + self.eps)

    def forward(self, g, h, node_type):
        graph_size  = g.batch_num_nodes(node_type) if self.is_node else g.batch_num_edges(node_type)
        x_list = torch.split(h, graph_size.tolist())
        norm_list = []
        for x in x_list:
            norm_list.append(self.norm(x))
        norm_x = torch.cat(norm_list, 0)

        if self.affine:
            return self.gamma * norm_x + self.beta
        else:
            return norm_x
        
from typing import Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.utils import scatter


def to_dense_batch_2D(x: Tensor, batch1: Tensor, batch2: Tensor, 
                   batch_all: Tensor, fill_value: float = 0., 
                   batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    r"""Given a sparse batch of node features
    :math:`\mathbf{X} \in \mathbb{R}^{(M_1 \times N_1 + \ldots + M_B \times N_B) \times F}` (with
    :math:`M_i` and :math:`N_i` indicating the number of nodes in graph :math:`i`), creates a
    dense node feature tensor
    :math:`\mathbf{X} \in \mathbb{R}^{B \times M_{\max} \times N_{\max} \times F}` (with
    :math:`M_{\max} = \max_i^B M_i`, :math:`N_{\max} = \max_i^B N_i`).
    In addition, a mask of shape :math:`\mathbf{M} \in \{ 0, 1 \}^{B \times
    M_{\max} \times N_{\max}}` is returned, holding information about the existence of
    fake-nodes in the dense representation.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch1 (LongTensor): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered.
        batch2 (LongTensor): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered.
        batch_all (LongTensor): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered.
        fill_value (float, optional): The value for invalid entries in the
            resulting dense output tensor. (default: :obj:`0`)
        batch_size (int, optional) The batch size. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`BoolTensor`)
    """

    if batch_size is None:
        batch_size = int(batch1.max()) + 1
        assert batch_size == int(batch2.max()) + 1

    num_nodes1 = scatter(batch1.new_ones(batch1.size(0)), batch1, dim=0,
                        dim_size=batch_size, reduce='sum')
    num_nodes2 = scatter(batch2.new_ones(batch2.size(0)), batch2, dim=0,
                        dim_size=batch_size, reduce='sum')
    num_nodes_all = num_nodes1 * num_nodes2
    cum_nodes_all = torch.cat([batch2.new_zeros(1), num_nodes_all.cumsum(dim=0)])

    max_num_nodes1 = int(num_nodes1.max())
    max_num_nodes2 = int(num_nodes2.max())

    tmp = torch.arange(x.size(0), device=x.device) - cum_nodes_all[batch_all]
    idx = tmp + (batch_all * max_num_nodes1 * max_num_nodes2)
    for i in range(len(cum_nodes_all)-1):
        tmp_i = torch.arange(num_nodes1[i], device=x.device).unsqueeze(1).repeat(1,num_nodes2[i]).view(-1)
        idx[cum_nodes_all[i]:cum_nodes_all[i+1]] += tmp_i * (max_num_nodes2 - num_nodes2[i])

    size = [batch_size * max_num_nodes1 * max_num_nodes2] + list(x.size())[1:]
    out = x.new_full(size, fill_value)
    out[idx] = x
    out = out.view([batch_size, max_num_nodes1, max_num_nodes2] + list(x.size())[1:])

    mask = torch.zeros(batch_size * max_num_nodes1 * max_num_nodes2, dtype=torch.bool,
                       device=x.device)
    mask[idx] = 1
    mask = mask.view(batch_size, max_num_nodes1, max_num_nodes2)

    return out, mask