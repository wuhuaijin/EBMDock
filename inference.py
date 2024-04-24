import torch
from config import *
from src.score_model import *
from torch.utils.data import DataLoader
from src.utils.eval import *
from src.data_process_one import *
from src.sampler import *


def set_random_seed():
    np.random.seed(args['random_seed'])
    torch.cuda.manual_seed_all(args['random_seed'])
    torch.manual_seed(args['random_seed'])
    print('random_seed:', args['random_seed'])
set_random_seed()
print(args)

# load cp model
args['bs'] = 1
args1 = args.copy()
args1['cp'] = True
load_cp_path = 'checkpoint/contact_prediction.pt'
checkpoint = torch.load(load_cp_path, map_location=args['device'])
model_cp = energy_model(args1).to(device)
model_cp.load_state_dict(checkpoint)
model_cp.eval()

# load energy model
args2 = args.copy()
args2['n_gaussians'] = 6
load_energy_path = 'checkpoint/energy.pt'
checkpoint = torch.load(load_energy_path, map_location=args['device'])
model_energy = energy_model(args2).to(device)
model_energy.load_state_dict(checkpoint)
model_energy.eval()

# energy_model_inference
inference_model = energy_model_inference(args).to(args['device'])
inference_model.eval()

sampler = SGLD(args)
meter = Meter_Unbound_Bound()
trans = Transformation(args).to(args['device'])

test_dataset = Inference_Dataset(args) # just one pair
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=docking_collate_fn_inf)


for i, data in enumerate(test_dataloader):

    pos_lig, pos_rec, atom_lig, atom_rec, batch_lig, batch_rec, lig_new_pos = data
    
    receptor_centor = pos_rec.mean(dim=0)
    pos_lig, pos_rec, atom_lig, atom_rec, batch_lig, batch_rec = pos_lig.to(device), pos_rec.to(device), \
            atom_lig.to(device), atom_rec.to(device), batch_lig.to(device), batch_rec.to(device)
            
    _, _, contact_info, bsp_pred = model_cp(pos_lig, batch_lig, atom_lig, pos_rec, batch_rec, atom_rec)
    

    cp_pred = contact_info[0].detach().cpu().view(-1)
    whole = pos_lig.shape[0] * pos_rec.shape[0]
    assert whole == cp_pred.shape[0]
    kcp = min(2.5 * (pos_lig.shape[0] + pos_rec.shape[0]), args['topk'])
    _, index_cp = cp_pred.topk(k=int(kcp))
    index_cp = index_cp.cpu().numpy()
    cp_mask = torch.zeros_like(cp_pred, dtype=torch.bool)
    cp_mask[index_cp] = True
    mdn_mask = torch.where(cp_mask)[0]
    
    gaussians_params, _, _, _ = model_energy(pos_lig, batch_lig, atom_lig, pos_rec, batch_rec, atom_rec, mdn_mask=mdn_mask)
    
    rotation_init, translation_init = sampler.random_sample(receptor_centor)    
    
    rotation_opt, translation_opt = sampler.sample(inference_model, rotation_init, translation_init, 
                                                   lig_new_pos, pos_rec, gaussians_params, mdn_mask, 
                                if_train=False, early_stop=False)
    
    negative_energy_all = []
    for i in range(args['sample_num']):
        R, t = rotation_opt.detach()[i], translation_opt.detach()[i]
        negative_energy = inference_model(lig_new_pos, pos_rec, gaussians_params, 
                                                R=R, t=t, mdn_mask=mdn_mask)
        
        negative_energy_all.append(negative_energy)
    
    best_negative_data_index = negative_energy_all.index(min(negative_energy_all))
    
    best_rotation = rotation_opt.detach()[best_negative_data_index]
    best_translation = translation_opt.detach()[best_negative_data_index]
    pred_lig_pos = trans(lig_new_pos, best_rotation, best_translation)[0]

    R_th, t_th = rigid_transform_Kabsch_3D(pos_lig.cpu().numpy().T, pred_lig_pos.cpu().numpy().T)
    AA, BB = pos_lig.cpu().numpy(), pred_lig_pos.cpu().numpy()

    assert  np.linalg.norm( ((R_th @ AA.T) + t_th ).T - BB) < 1e-1

    ppdb_ligand = PandasPdb().read_pdb(args['ligand_file_path'])

    unbound_ligand_all_atoms_pre_pos = ppdb_ligand.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32)

    ppdb_ligand.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = ((R_th @ unbound_ligand_all_atoms_pre_pos.T) + t_th).T

    ppdb_ligand.to_pdb(path=args['output_ligand_path'], records=['ATOM'], gz=False)