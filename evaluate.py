import torch
from config import *
from src.score_model import *
from torch.utils.data import DataLoader
from src.utils.eval import *
from src.data_loader import *
from src.sampler import *


def set_random_seed():
    np.random.seed(args['random_seed'])
    torch.cuda.manual_seed_all(args['random_seed'])
    torch.manual_seed(args['random_seed'])
    print('random_seed:', args['random_seed'])
set_random_seed()


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
args2['intersection_loss'] = True
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

test_dataset =  DockingDataset(args, args['dataset'], reload_mode='test', load_from_cache=False, raw_data_path=args['data_path'])
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=docking_collate_fn)
     

for i, data in enumerate(test_dataloader):

    pos_lig, pos_rec, atom_lig, atom_rec, batch_lig, batch_rec, bsp_lig, bsp_rec, file_name, lig_new_pos, RT = data
        
    receptor_centor = pos_rec.mean(dim=0).unsqueeze(0)
    
    pos_lig, pos_rec, atom_lig, atom_rec, batch_lig, batch_rec, bsp_lig, bsp_rec = pos_lig.to(device),\
        pos_rec.to(device), atom_lig.to(device), atom_rec.to(device), batch_lig.to(device), \
            batch_rec.to(device), bsp_lig.to(device), bsp_rec.to(device)
    
    _, _, contact_info, bsp_pred = model_cp(pos_lig, batch_lig, atom_lig, pos_rec, batch_rec, atom_rec)

    if args['known_interface']:
        cp_truth = contact_info[1].detach().cpu().view(-1)
        cp_truth = torch.where(cp_truth > 0, 1, 0)            
        mdn_mask = torch.where(cp_truth)[0]
    else:
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
    positive_energy = inference_model(pos_lig, pos_rec, gaussians_params, 
                                                R=None, t=None, mdn_mask=mdn_mask)    
    negative_energy_all = []
    for i in range(args['sample_num']):
        R, t = rotation_opt.detach()[i], translation_opt.detach()[i]
        negative_energy = inference_model(lig_new_pos, pos_rec, gaussians_params, 
                                                R=R, t=t, mdn_mask=mdn_mask)
        negative_energy_all.append(negative_energy)
        pred_lig_pos = trans(lig_new_pos, R, t)[0]
    
    best_negative_data_index = negative_energy_all.index(min(negative_energy_all))
    
    best_rotation = rotation_opt.detach()[best_negative_data_index]
    best_translation = translation_opt.detach()[best_negative_data_index]
    pred_lig_pos = trans(lig_new_pos, best_rotation, best_translation)[0]
    _, dockq = meter.update_rmsd(pred_lig_pos, pos_rec, pos_lig, pos_rec)
    print(dockq)


ligand_rmsd_mean, complex_rmsd_mean, interface_rmsd_mean, bad_case = meter.summarize(reduction_rmsd='mean')
ligand_rmsd_median, complex_rmsd_median, interface_rmsd_median, bad_case = meter.summarize(reduction_rmsd='median')
ligand_rmsd_std, complex_rmsd_std, interface_rmsd_std, bad_case = meter.summarize(reduction_rmsd='std')
dockQ_mean, dockQ_median, dockQ_std = meter.dockQ()


print('[Test] --> || lrmsd loss mean/median/std {:.4f}/{:.4f}/{:.4f} \
            || crmsd loss mean/median/std {:.4f}/{:.4f}/{:.4f} || irmsd loss mean/median/std {:.4f}/{:.4f}/{:.4f} \
            || dockQ mean/median/std {:.4f}/{:.4f}/{:.4f} '.format(
            ligand_rmsd_mean, ligand_rmsd_median, ligand_rmsd_std, 
            complex_rmsd_mean, complex_rmsd_median, complex_rmsd_std,
            interface_rmsd_mean, interface_rmsd_median, interface_rmsd_std,
            dockQ_mean, dockQ_median, dockQ_std))


with open('result.txt', 'a') as f:
    f.write('sgld\n')
    f.write(str(args['intersection_loss_weight'])+ '\n')
    res = '[Test] --> || lrmsd loss mean/median/std {:.4f}/{:.4f}/{:.4f} \
    || crmsd loss mean/median/std {:.4f}/{:.4f}/{:.4f} || irmsd loss mean/median/std {:.4f}/{:.4f}/{:.4f} \
    || dockQ mean/median/std {:.4f}/{:.4f}/{:.4f} \n'.format(
    ligand_rmsd_mean, ligand_rmsd_median, ligand_rmsd_std, 
    complex_rmsd_mean, complex_rmsd_median, complex_rmsd_std,
    interface_rmsd_mean, interface_rmsd_median, interface_rmsd_std,
    dockQ_mean, dockQ_median, dockQ_std)
    f.write(res)
    
    
    
