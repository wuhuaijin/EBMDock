import argparse

import torch


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser()


# add parameters
# data process
parser.add_argument('-translation_interval', default=10.0, type=float, required=False)
parser.add_argument('-bsp_threshold', default=8.0, type=float, required=False)

# training
parser.add_argument('-bs', default=4, type=int, required=False)
parser.add_argument('-lr', default=1e-4, type=float, required=False)
parser.add_argument('-n_epochs', default=50, type=int, required=False)
parser.add_argument('-random_seed', default=123, type=int, required=False)
parser.add_argument('-energy_temp', default=0.1, type=float, required=False)
parser.add_argument('-dataset', default='dipshet', type=str, required=False) # db5
parser.add_argument('-data_path', default='./data/DIPS-Het', type=str, required=False)


# save_model and log
parser.add_argument('-save_path_energy', type=str, default='./save/', help='Path to save the file')
parser.add_argument('-save_path_prediction', type=str, default='./save/', help='Path to save the file')
parser.add_argument('-log_file', type=str, default='example.log', help='Log file name')

# task
parser.add_argument('-bsp', default=False, required=False, action='store_true')
parser.add_argument('-ebm_train', default=False, required=False, action='store_true')
parser.add_argument('-ll_train', default=False, required=False, action='store_true') # use loglikelihood to train the model
parser.add_argument('-cp', default=False, required=False, action='store_true')
parser.add_argument('-cp_alpha', default=0.07, required=False, type=float)
parser.add_argument('-n_gaussians', default=0, type=int, required=False)
parser.add_argument('-bsgt4mdn', default=False, required=False, action='store_true')
parser.add_argument('-dist_threshold', default=100, required=False, type=float)
parser.add_argument('-contact_threshold', default=15, required=False, type=float)
parser.add_argument('-topk', default=1500, required=False, type=int)
parser.add_argument('-fine_tune', default=False, required=False, action='store_true')


# network
parser.add_argument('-equiformer_out_dim', default=128, type=int, required=False)
parser.add_argument('-dropout_rate', default=0, type=float, required=False)
parser.add_argument('-hidden_dim', default=128, type=int, required=False)
parser.add_argument('-cross_attention', default=False, required=False, action='store_true')


# intersection loss 
parser.add_argument('-intersection_loss', default=False, required=False, action='store_true')
parser.add_argument('-intersection_loss_weight', type=float, default=5, required=False)
parser.add_argument('-intersection_sigma', type=float, default=8., required=False) #25
parser.add_argument('-intersection_surface_ct', type=float, default=8., required=False) #10

# sgld
parser.add_argument('-sgld_lr', default=0.01, type=float, required=False)
parser.add_argument('-sgld_R_var', default=0.01, type=float, required=False)
parser.add_argument('-sgld_R_clip', default=1.0, type=float, required=False)
parser.add_argument('-sgld_T_var', default=0.1, type=float, required=False)
parser.add_argument('-sgld_T_clip', default=1.0, type=float, required=False)
parser.add_argument('-sample_iters', default=100, type=int, required=False)  # 50 for training, 100 for inference
parser.add_argument('-sample_num', default=5, type=int, required=False)
parser.add_argument('-max_translation', default=500.0, type=float, required=False, help='max translation range')
parser.add_argument('-translation_std_rec', default=40.0, type=float, required=False, help='random translation variance from receptor')
parser.add_argument('-translation_std_lig', default=3.0, type=float, required=False, help='random translation variance from ligand')
parser.add_argument('-select_p', default=1.0, type=float, required=False, help='the probability of initial sample data from normal distribution with center at receptor; 1-p at ligand')

# inference
parser.add_argument('-receptor_file_path', default=None, type=str, required=False)
parser.add_argument('-ligand_file_path', default=None, type=str, required=False)
parser.add_argument('-output_ligand_path', default='output_ligand.pdb', type=str, required=False)
parser.add_argument('-known_interface', default=False, required=False, action='store_true')

args = parser.parse_args().__dict__

args['device'] = device


if args['ebm_train'] or args['ll_train']:
    assert args['n_gaussians'] > 0