import torch
import e3nn
from e3nn import o3
from .nets.graph_attention_transformer import NodeEmbeddingNetwork, GaussianRadialBasisLayer,\
    RadialBasis, EdgeDegreeEmbeddingNetwork,get_norm_layer,EquivariantDropout,LinearRS,Activation,TransBlock,\
        EquivariantLayerNormV2,EquivariantInstanceNorm,EquivariantGraphNorm
from torch_cluster import radius_graph
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F
from torch.distributions import Normal
from torch_scatter import scatter_mean
from .utils.test_utils import compute_body_intersection_loss, Transformation

_MAX_ATOM_TYPE = 21
_RESCALE = True
_USE_BIAS = True
_AVG_DEGREE = 500

class GraphAttentionTransformer(torch.nn.Module):
    def __init__(self,
        irreps_in='5x0e',
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=5.0,
        number_of_basis=128, basis_type='gaussian', fc_neurons=[64, 64], 
        irreps_feature='512x0e',
        irreps_feature_out='128x0e',
        irreps_head='32x0e+16x1o+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=False,
        irreps_mlp_mid='128x0e+64x1e+32x2e',
        norm_layer='layer',
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0,
        drop_path_rate=0.0,
        mean=None, std=None, scale=None, atomref=None, cross_attention=False):
        super().__init__()

        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.task_mean = mean
        self.task_std = std
        self.scale = scale
        self.register_buffer('atomref', atomref)

        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_input = o3.Irreps(irreps_in)
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.lmax = self.irreps_node_embedding.lmax
        self.irreps_feature = o3.Irreps(irreps_feature)
        self.irreps_feature_out = o3.Irreps(irreps_feature_out)
        self.num_layers = num_layers
        self.irreps_edge_attr = o3.Irreps(irreps_sh) if irreps_sh is not None \
            else o3.Irreps.spherical_harmonics(self.lmax)
        self.fc_neurons = [self.number_of_basis] + fc_neurons
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.irreps_pre_attn = irreps_pre_attn
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid)
        
        self.atom_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _MAX_ATOM_TYPE)
        self.basis_type = basis_type
        if self.basis_type == 'gaussian':
            self.rbf = GaussianRadialBasisLayer(self.number_of_basis, cutoff=self.max_radius)
        elif self.basis_type == 'bessel':
            self.rbf = RadialBasis(self.number_of_basis, cutoff=self.max_radius, 
                rbf={'name': 'spherical_bessel'})
        else:
            raise ValueError
        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(self.irreps_node_embedding, 
            self.irreps_edge_attr, self.fc_neurons, _AVG_DEGREE)
        
        self.blocks = torch.nn.ModuleList()
        
        self.cross_attention = cross_attention
        self.build_blocks()
        
        self.norm = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.out_dropout = None
        if self.out_drop != 0.0:
            self.out_dropout = EquivariantDropout(self.irreps_feature, self.out_drop)
        self.head = torch.nn.Sequential(
            LinearRS(self.irreps_feature, self.irreps_feature, rescale=_RESCALE), 
            Activation(self.irreps_feature, acts=[torch.nn.SiLU()]),
            LinearRS(self.irreps_feature, self.irreps_feature_out, rescale=_RESCALE)) 
        # self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)
        self.apply(self._init_weights)
        
        
        
    def build_blocks(self):
        for i in range(self.num_layers):
            if i != (self.num_layers - 1):
                irreps_block_output = self.irreps_node_embedding
            else:
                irreps_block_output = self.irreps_feature
            blk = TransBlock(irreps_node_input=self.irreps_node_embedding, 
                irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=self.irreps_edge_attr, 
                irreps_node_output=irreps_block_output,
                fc_neurons=self.fc_neurons, 
                irreps_head=self.irreps_head, 
                num_heads=self.num_heads, 
                irreps_pre_attn=self.irreps_pre_attn, 
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                alpha_drop=self.alpha_drop, 
                proj_drop=self.proj_drop,
                drop_path_rate=self.drop_path_rate,
                irreps_mlp_mid=self.irreps_mlp_mid,
                norm_layer=self.norm_layer,
                cross_attention=self.cross_attention)
            self.blocks.append(blk)
            
            
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
            
                          
    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Linear) 
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormV2)
                or isinstance(module, EquivariantInstanceNorm)
                or isinstance(module, EquivariantGraphNorm)
                or isinstance(module, GaussianRadialBasisLayer) 
                or isinstance(module, RadialBasis)):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear) and 'weight' in parameter_name:
                        continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
                    
        return set(no_wd_list)
        

    def forward(self, f_in, pos1, batch1, node_atom1, pos2, batch2, node_atom2, **kwargs) -> torch.Tensor:
        
        edge_src1, edge_dst1 = radius_graph(pos1, r=self.max_radius, batch=batch1,
            max_num_neighbors=1000)
        edge_vec1 = pos1.index_select(0, edge_src1) - pos1.index_select(0, edge_dst1)
        edge_sh1 = o3.spherical_harmonics(l=self.irreps_edge_attr,
            x=edge_vec1, normalize=True, normalization='component')
        
        # node_atom1 = node_atom1.new_tensor([-1, 0, -1, -1, -1, -1, 1, 2, 3, 4])[node_atom1]
        atom_embedding1, atom_attr1, atom_onehot1 = self.atom_embed(node_atom1)
        edge_length1 = edge_vec1.norm(dim=1)
        #edge_length_embedding = sin_pos_embedding(x=edge_length, 
        #    start=0.0, end=self.max_radius, number=self.number_of_basis, 
        #    cutoff=False)
        edge_length_embedding1 = self.rbf(edge_length1)
        edge_degree_embedding1 = self.edge_deg_embed(atom_embedding1, edge_sh1, 
            edge_length_embedding1, edge_src1, edge_dst1, batch1)
        node_features1 = atom_embedding1 + edge_degree_embedding1
        node_attr1 = torch.ones_like(node_features1.narrow(1, 0, 1))
        
        
        edge_src2, edge_dst2 = radius_graph(pos2, r=self.max_radius, batch=batch2,
            max_num_neighbors=1000)
        edge_vec2 = pos2.index_select(0, edge_src2) - pos2.index_select(0, edge_dst2)
        edge_sh2 = o3.spherical_harmonics(l=self.irreps_edge_attr,
            x=edge_vec2, normalize=True, normalization='component')
        
        # node_atom2 = node_atom2.new_tensor([-1, 0, -1, -1, -1, -1, 1, 2, 3, 4])[node_atom2]
        atom_embedding2, atom_attr2, atom_onehot2 = self.atom_embed(node_atom2)
        edge_length2 = edge_vec2.norm(dim=1)
        #edge_length_embedding = sin_pos_embedding(x=edge_length, 
        #    start=0.0, end=self.max_radius, number=self.number_of_basis, 
        #    cutoff=False)
        edge_length_embedding2 = self.rbf(edge_length2)
        edge_degree_embedding2 = self.edge_deg_embed(atom_embedding2, edge_sh2, 
            edge_length_embedding2, edge_src2, edge_dst2, batch2)
        node_features2 = atom_embedding2 + edge_degree_embedding2
        node_attr2 = torch.ones_like(node_features2.narrow(1, 0, 1))
        
        
        for blk in self.blocks:
            node_features1, node_features2 = blk(node_input1=node_features1, node_attr1=node_attr1, 
                edge_src1=edge_src1, edge_dst1=edge_dst1, edge_attr1=edge_sh1, 
                edge_scalars1=edge_length_embedding1, 
                batch1=batch1,
                node_input2=node_features2, node_attr2=node_attr2, 
                edge_src2=edge_src2, edge_dst2=edge_dst2, edge_attr2=edge_sh2, 
                edge_scalars2=edge_length_embedding2, 
                batch2=batch2)
        
        node_features2 = self.norm(node_features2, batch=batch2)
        if self.out_dropout is not None:
            node_features2 = self.out_dropout(node_features2)
        outputs2 = self.head(node_features2)
        # outputs2 = self.scale_scatter(outputs2, batch2, dim=0)
        
        node_features1 = self.norm(node_features1, batch=batch1)
        if self.out_dropout is not None:
            node_features1 = self.out_dropout(node_features1)
        outputs1 = self.head(node_features1)
        
        if self.scale is not None:
            outputs2 = self.scale * outputs2
            outputs1 = self.scale * outputs1

        return outputs1, outputs2
    
    
class distance_ll(torch.nn.Module):
    
    def __init__(self, args):
        super(distance_ll, self).__init__()
        self.args = args
        hidden_dim = args['hidden_dim']
        dropout_rate = args['dropout_rate']
        self.device = args['device']
        n_gaussians = args['n_gaussians']
        self.dist_threshold = args['dist_threshold']
        feature_out = args['equiformer_out_dim']
        self.contact_threshold = args['contact_threshold']
        
        
        self.MLP = torch.nn.Sequential(torch.nn.Linear(2*feature_out, hidden_dim), 
                                 torch.nn.BatchNorm1d(hidden_dim), torch.nn.ELU(), 
                                 torch.nn.Dropout(p=dropout_rate))
        self.z_pi = torch.nn.Linear(hidden_dim, n_gaussians)
        self.z_sigma = torch.nn.Linear(hidden_dim, n_gaussians)
        self.z_mu = torch.nn.Linear(hidden_dim, n_gaussians)
        
        
        self.contact_prediction = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, 1),
            )
        if self.args['ebm_train']:
            self.dist_threshold = self.args['contact_threshold']
    
    def mdn_loss_fn(self, logpi, sigma, mu, y):
        normal = Normal(mu, sigma)
        loglik = normal.log_prob(y.expand_as(normal.loc))
        loss = -torch.logsumexp(logpi + loglik, dim=1)

        return loss
        
        
    def forward(self, feature_lig, pos_lig, batch_lig, feature_rec, pos_rec, batch_rec, mdn_mask=None):
        h_l_x, l_mask = to_dense_batch(feature_lig, batch_lig, fill_value=0)
        h_r_x, r_mask = to_dense_batch(feature_rec, batch_rec, fill_value=0)
        
        (B, N_l, C_out), N_r = h_l_x.size(), h_r_x.size(1)
        C_mask = l_mask.view(B, N_l, 1) & r_mask.view(B, 1, N_r)
        
        C_batch = torch.tensor(range(B)).unsqueeze(-1).unsqueeze(-1)
        C_batch = C_batch.to(self.device)
        C_batch = C_batch.repeat(1, N_l, N_r)[C_mask]
        
        h_l_pos, _ = to_dense_batch(pos_lig, batch_lig, fill_value=0)
        h_r_pos, _ = to_dense_batch(pos_rec, batch_rec, fill_value=0)
        pair_dis = torch.cdist(h_l_pos, h_r_pos, compute_mode='donot_use_mm_for_euclid_dist')[C_mask]
        pair_dis = pair_dis.unsqueeze(dim=1)
        
        h_l_x = h_l_x.unsqueeze(-2)
        h_l_x = h_l_x.repeat(1, 1, N_r, 1) # [B, N_l, N_r, C_out]

        h_r_x = h_r_x.unsqueeze(-3)
        h_r_x = h_r_x.repeat(1, N_l, 1, 1) # [B, N_l, N_r, C_out]
        C = torch.cat((h_l_x, h_r_x), -1) # (B, N_l, N_r, 2*C_out)
        C = C[C_mask]# (N_c, 2*C_out)
        
        C = self.MLP(C)
        
        contact_pred_out, contact_truth = None, None
        if self.args['cp']:
            contact_pred_out = self.contact_prediction(C)
        contact_truth = torch.where(pair_dis < self.contact_threshold, 1, 0)
            
        if mdn_mask is None:
            mdn_mask = torch.where(pair_dis < self.dist_threshold)[0]
        C_mdn = C[mdn_mask]
        
        logpi = F.log_softmax(self.z_pi(C_mdn), -1)
        sigma = F.elu(self.z_sigma(C_mdn))+1.1
        mu = F.elu(self.z_mu(C_mdn))+1
        dist = pair_dis[mdn_mask]
        
        gaussian_params = [logpi, sigma, mu, C_mask, batch_lig, batch_rec]  
        mdn = self.mdn_loss_fn(logpi, sigma, mu, dist)

        prob = scatter_mean(mdn, C_batch[mdn_mask], dim=0, dim_size=C_batch.unique().size(0))
                
        return gaussian_params, prob, [contact_pred_out, contact_truth]
    

class energy_model(torch.nn.Module):
    def __init__(self, args):
        super(energy_model, self).__init__()
        
        self.out_feat_dim = str(args['equiformer_out_dim']) + 'x0e'
        self.encoder = GraphAttentionTransformer(
            irreps_in='5x0e',
            irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
            irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
            max_radius=8.0,
            number_of_basis=128, fc_neurons=[64, 64], 
            irreps_feature='512x0e',irreps_feature_out=self.out_feat_dim,
            irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
            rescale_degree=False, nonlinear_message=True,
            irreps_mlp_mid='384x0e+192x1e+96x2e',
            norm_layer='layer',
            alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
            mean=None, std=None, scale=None, atomref=None, cross_attention=args['cross_attention'])
        self.args = args
        if self.args['n_gaussians'] > 0 or self.args['cp']:
            self.distance = distance_ll(args)
        
        feature_out = args['equiformer_out_dim']
        dropout_rate = args['dropout_rate']
        hidden_dim = args['hidden_dim']
        
        self.bsp = torch.nn.Sequential(
                torch.nn.Linear(feature_out, hidden_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, 1),
            )
        self.trans = Transformation(args)
        
    def forward(self, pos1, batch1, node_atom1, pos2, batch2, node_atom2, mdn_mask=None):
        
        outputs1, outputs2 = self.encoder(None, pos1, batch1, node_atom1, pos2, batch2, node_atom2)
        prob, gaussian_params = None, None
        bsp_lig_pred, bsp_rec_pred, contact_pred, contact_truth = None, None, None, None
        
        if self.args['n_gaussians'] > 0 or self.args['cp']:
            gaussian_params, prob, [contact_pred, contact_truth] = self.distance(outputs1, pos1, batch1, outputs2, pos2, batch2, mdn_mask)
        
        if self.args['bsp']:
            bsp_lig_pred, bsp_rec_pred = self.bsp(outputs1), self.bsp(outputs2)
                
        return gaussian_params, prob, [contact_pred, contact_truth], [bsp_lig_pred, bsp_rec_pred]
    

class energy_model_inference(torch.nn.Module):
    def __init__(self, args):
        super(energy_model_inference, self).__init__()
        self.trans = Transformation(args)
        self.args = args
    
    def mdn_loss_fn(self, logpi, sigma, mu, y):
        normal = Normal(mu, sigma)
        loglik = normal.log_prob(y.expand_as(normal.loc))
        loss = -torch.logsumexp(logpi + loglik, dim=1)

        return loss
    
    def compute_euclidean_distances_matrix(self, X, Y):
        # Based on: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
        # (X-Y)^2 = X^2 + Y^2 -2XY
        X = X.double()
        Y = Y.double()
        dists = -2 * torch.bmm(X, Y.permute(0, 2, 1)) + torch.sum(Y**2,    axis=-1).unsqueeze(1) + torch.sum(X**2, axis=-1).unsqueeze(-1)
        return dists**0.5
    
    def forward(self, pos1, pos2, guassian_params, R=None, t=None, mdn_mask=None):
        
        if R is not None and t is not None:
            pos1 = self.trans(pos1, R, t)
            pos1 = torch.cat(pos1, dim=0)
        
        logpi, sigma, mu, C_mask, batch_lig, batch_rec = guassian_params
        
        h_l_pos, _ = to_dense_batch(pos1, batch_lig, fill_value=0)
        h_r_pos, _ = to_dense_batch(pos2, batch_rec, fill_value=0)
        pair_dis = torch.cdist(h_l_pos, h_r_pos, compute_mode='donot_use_mm_for_euclid_dist')[C_mask]
        pair_dis = pair_dis.unsqueeze(dim=1)
        dist = pair_dis[mdn_mask]
                
        mdn = self.mdn_loss_fn(logpi, sigma, mu, dist)
        mdn = mdn.mean()
        
        if self.args['intersection_loss']:
            assert self.args['bs'] == 1
            intersection_loss = compute_body_intersection_loss(pos1, pos2, sigma=self.args['intersection_sigma'], surface_ct=self.args['intersection_surface_ct'])
        else:
            intersection_loss = 0
        
        return mdn + intersection_loss * self.args['intersection_loss_weight']
    
class model_sampler(torch.nn.Module):
    def __init__(self, args):
        super(model_sampler, self).__init__()
        self.trans = Transformation(args)
        self.args = args
    
    def mdn_loss_fn(self, logpi, sigma, mu, y):
        normal = Normal(mu, sigma)
        loglik = normal.log_prob(y.expand_as(normal.loc))
        loss = -torch.logsumexp(logpi + loglik, dim=1)

        return loss
    
    def compute_euclidean_distances_matrix(self, X, Y):
        # Based on: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
        # (X-Y)^2 = X^2 + Y^2 -2XY
        X = X.double()
        Y = Y.double()
        dists = -2 * torch.bmm(X, Y.permute(0, 2, 1)) + torch.sum(Y**2,    axis=-1).unsqueeze(1) + torch.sum(X**2, axis=-1).unsqueeze(-1)
        return dists**0.5
    
    def forward(self, pos1, pos2, guassian_params, R=None, t=None, mdn_mask=None):
        
        if R is not None and t is not None:

            pos1 = self.trans(pos1, R, t)
            pos1 = torch.cat(pos1, dim=0)
        
        logpi, sigma, mu, C_mask, batch_lig, batch_rec = guassian_params
        
        h_l_pos, _ = to_dense_batch(pos1, batch_lig, fill_value=0)
        h_r_pos, _ = to_dense_batch(pos2, batch_rec, fill_value=0)
        pair_dis = torch.cdist(h_l_pos, h_r_pos, compute_mode='donot_use_mm_for_euclid_dist')[C_mask]
        pair_dis = pair_dis.unsqueeze(dim=1)
        dist = pair_dis[mdn_mask]
                
        mdn = self.mdn_loss_fn(logpi, sigma, mu, dist)
        mdn = mdn.mean()
                
        return mdn

    
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha])
        self.gamma = gamma
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, bsp, iface_label):
        bce_loss = self.bce_loss(bsp, iface_label.float())
        iface_label = iface_label.long()
        at = self.alpha.to(bsp.device).gather(0, iface_label.data.view(-1))
        pt = torch.exp(-bce_loss)
        focal_loss = at * (1-pt)**self.gamma * bce_loss

        return focal_loss.mean()
    
    

