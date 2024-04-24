import torch
from src.data_loader import *
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader
from src.score_model import *
from torchmetrics.functional import average_precision
from torchmetrics import AUROC, Precision
from config import *
from src.sampler import *
from src.utils.eval import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
sampler = SGLD(args)

def ContrastiveDivergence(args, positive_energy, negative_energy):
    pos_loss = torch.mean(positive_energy)
    neg_loss = torch.mean(negative_energy)
    loss_ml =  (pos_loss - neg_loss) + args['energy_temp'] * torch.mean(positive_energy**2 + negative_energy**2)

    return loss_ml

def val(data_loader, model, sample_model, args, log_file):
    
    model.eval()
    
    bsp_pred_epoch = torch.empty(0)
    bsp_truth_epoch = torch.empty(0, dtype=torch.int64)
    trans = Transformation(args).to(args['device'])
    meter = Meter_Unbound_Bound()
    
    auroc = AUROC(task="binary")
    precision = Precision(task="binary")
    mdn_loss_val = 0.0
    APs_cp, AUCs_cp, Ps_cp = [], [], []
    
    for id, data in enumerate(data_loader):
        
        pos_lig, pos_rec, atom_lig, atom_rec, batch_lig, batch_rec, bsp_lig, bsp_rec, file_name, lig_new_pos, _ = data
        ligand_center = []
        receptor_center = []
        num_nodes_lig = torch.bincount(batch_lig).tolist()
        num_nodes_rec = torch.bincount(batch_rec).tolist()

        pos_lig_list = torch.split(pos_lig, num_nodes_lig)
        pos_rec_list = torch.split(pos_rec, num_nodes_rec)
       
        for i in range(len(num_nodes_lig)):
            ligand_center.append(torch.mean(pos_lig_list[i], dim=0))
            receptor_center.append(torch.mean(pos_rec_list[i], dim=0))
        ligand_center = torch.stack(ligand_center)
        receptor_center = torch.stack(receptor_center)


        pos_lig, pos_rec, atom_lig, atom_rec, batch_lig, batch_rec, bsp_lig, bsp_rec = pos_lig.to(device),\
                pos_rec.to(device), atom_lig.to(device), atom_rec.to(device), batch_lig.to(device), \
                    batch_rec.to(device), bsp_lig.to(device), bsp_rec.to(device)        
        
        gaussians_params, positive_energy, contact_info, bsp_pred = model(pos_lig, batch_lig, atom_lig, pos_rec, batch_rec, atom_rec)
        
        if args['n_gaussians'] > 0 and args['ebm_train']:
            cp_truth = contact_info[1].detach().cpu().view(-1)
            if (cp_truth > 0).any():
                cp_truth = torch.where(cp_truth > 0, 1, 0)
                mdn_mask = torch.where(cp_truth)[0]
            
            rotation_init, translation_init = sampler.random_sample(receptor_center) 
            rotation_opt, translation_opt = sampler.sample(sample_model, rotation_init, translation_init, 
                                                        lig_new_pos, pos_rec, gaussians_params, mdn_mask, 
                                        if_train=True, early_stop=False)
            
            negative_energy_all = []
            for i in range(args['sample_num']):
                R, t = rotation_opt.detach()[i], translation_opt.detach()[i]
                negative_energy = sample_model(lig_new_pos, pos_rec, gaussians_params, 
                                                        R=R, t=t, mdn_mask=mdn_mask)
                negative_energy_all.append(negative_energy)
            negative_energy_all = torch.stack(negative_energy_all)
            negative_energy = torch.mean(negative_energy_all, dim=0)
            _, best_negative_data_index = torch.min(negative_energy_all, dim=0)
            
            best_rotation = rotation_opt.detach()[best_negative_data_index, torch.arange(len(receptor_center)), :]
            best_translation = translation_opt.detach()[best_negative_data_index, torch.arange(len(receptor_center)), :]
            pred_lig_pos = trans(lig_new_pos, best_rotation, best_translation)
            for i in range(len(pred_lig_pos)):
                meter.update_rmsd(pred_lig_pos[i], pos_rec_list[i], pos_lig_list[i], pos_rec_list[i])
            
        
        if args['n_gaussians'] > 0 and args['ll_train']:
            mdn_loss_val = positive_energy.mean().item()
            
        
        if args['bsp']:
            bsp_pred_epoch = torch.cat((bsp_pred_epoch, bsp_pred[0].detach().cpu(), bsp_pred[1].detach().cpu()), dim = 0)
            bsp_truth_epoch = torch.cat((bsp_truth_epoch, bsp_lig.cpu(), bsp_rec.cpu()), dim = 0)
            
        if args['cp']:
            A = contact_info[0].squeeze().detach().cpu()
            _, index_cp = A.topk(k=500)
            result = torch.zeros_like(A)
            result[index_cp] = 1

            P_cp = precision(result, contact_info[1].squeeze().detach().cpu())
            AP_cp = average_precision(contact_info[0].squeeze().detach().cpu(), contact_info[1].squeeze().detach().cpu(), task="binary")
            AUC_cp = auroc(contact_info[0].squeeze().detach().cpu(), contact_info[1].squeeze().detach().cpu())
            APs_cp.append(AP_cp)
            AUCs_cp.append(AUC_cp)
            Ps_cp.append(P_cp)
            
    P = None
     
    if args['bsp']:   
        P = precision(bsp_pred_epoch.squeeze(), bsp_truth_epoch)
        AP = average_precision(bsp_pred_epoch.squeeze(), bsp_truth_epoch, task="binary")
        AUC = auroc(bsp_pred_epoch.squeeze(), bsp_truth_epoch)
        print('Valid P', P, 'Valid AP:', AP, 'Valid AUC:', AUC)
        log_file.write('Valid P' + str(P) + 'Valid AP:' + str(AP) + 'Valid AUC:' + str(AUC) + '\n')

    if args['cp']:
        P = sum(Ps_cp) / len(Ps_cp)
        AP = sum(APs_cp) / len(APs_cp)
        AUC = sum(AUCs_cp) / len(AUCs_cp)
        print('Valid cp P:', P, 'Valid cp AP:', AP, 'Valid cp AUC:', AUC)
        log_file.write('Valid cp P:' + str(P) + 'Valid cp AP:' + str(AP) + 'Valid cp AUC:' + str(AUC) + '\n')  
    
    if args['n_gaussians'] > 0 and args['ebm_train']:
        ligand_rmsd_mean, complex_rmsd_mean, interface_rmsd_mean, bad_case = meter.summarize(reduction_rmsd='mean')
        print('ligand_rmsd_mean:', ligand_rmsd_mean, 'complex_rmsd_mean:', complex_rmsd_mean, 'interface_rmsd_mean:', interface_rmsd_mean)
        return P, ligand_rmsd_mean
    
    if args['n_gaussians'] > 0 and args['ll_train']:
        return P, mdn_loss_val / (id + 1)
    


def train_one_epoch(data_loader, model, sample_model, bsp_loss_func, cp_loss_func, optimizer, args, log_file):
    
    bsp_pred_epoch = torch.empty(0)
    bsp_truth_epoch = torch.empty(0, dtype=torch.int64)
    
    auroc = AUROC(task="binary")
    
    bsp_loss_epoch, cd_loss_epoch, mdn_loss_epoch, cp_loss_epoch = 0.0, 0.0, 0.0, 0.0
    APs_cp, AUCs_cp = [], []
    
    model.train()
    
    for id, data in enumerate(data_loader):
        
        optimizer.zero_grad()
        pos_lig, pos_rec, atom_lig, atom_rec, batch_lig, batch_rec, bsp_lig, bsp_rec, file_name, lig_new_pos, _ = data
        ligand_center = []
        receptor_center = []
        num_nodes_lig = torch.bincount(batch_lig).tolist()
        num_nodes_rec = torch.bincount(batch_rec).tolist()

        pos_lig_list = torch.split(pos_lig, num_nodes_lig)
        pos_rec_list = torch.split(pos_rec, num_nodes_rec)
       
        for i in range(len(num_nodes_lig)):
            ligand_center.append(torch.mean(pos_lig_list[i], dim=0))
            receptor_center.append(torch.mean(pos_rec_list[i], dim=0))
        ligand_center = torch.stack(ligand_center)
        receptor_center = torch.stack(receptor_center)


        pos_lig, pos_rec, atom_lig, atom_rec, batch_lig, batch_rec, bsp_lig, bsp_rec = pos_lig.to(device),\
                pos_rec.to(device), atom_lig.to(device), atom_rec.to(device), batch_lig.to(device), \
                    batch_rec.to(device), bsp_lig.to(device), bsp_rec.to(device)        
        loss = torch.zeros([]).to(args['device'])
        
        gaussians_params, positive_energy, contact_info, bsp_pred = model(pos_lig, batch_lig, atom_lig, pos_rec, batch_rec, atom_rec)
        
        if args['n_gaussians'] > 0 and args['ebm_train']:
            cp_truth = contact_info[1].detach().cpu().view(-1)
            if (cp_truth > 0).any():
                cp_truth = torch.where(cp_truth > 0, 1, 0)
                mdn_mask = torch.where(cp_truth)[0]
            
            rotation_init, translation_init = sampler.random_sample(receptor_center) 
            rotation_opt, translation_opt = sampler.sample(sample_model, rotation_init, translation_init, 
                                                        lig_new_pos, pos_rec, gaussians_params, mdn_mask, 
                                        if_train=True, early_stop=False)
            
            negative_energy_all = []
            for i in range(args['sample_num']):
                R, t = rotation_opt.detach()[i], translation_opt.detach()[i]
                negative_energy = sample_model(lig_new_pos, pos_rec, gaussians_params, 
                                                        R=R, t=t, mdn_mask=mdn_mask)
                negative_energy_all.append(negative_energy)
            negative_energy_all = torch.stack(negative_energy_all)
            negative_energy = torch.mean(negative_energy_all, dim=0)

            cd_loss = ContrastiveDivergence(args, positive_energy, negative_energy)
            cd_loss_epoch += cd_loss.item()
            loss += cd_loss
        
        if args['n_gaussians'] > 0 and args['ll_train']:
            mdn_loss = positive_energy.mean()
            mdn_loss_epoch += mdn_loss.item()
            loss += mdn_loss
        
        if args['bsp']:
            bsp_loss = bsp_loss_func(torch.cat((bsp_pred[0].squeeze(), bsp_pred[1].squeeze()), dim = 0), torch.cat((bsp_lig, bsp_rec), dim = 0))
            bsp_loss_epoch += bsp_loss.item()
            loss += bsp_loss
            bsp_pred_epoch = torch.cat((bsp_pred_epoch, bsp_pred[0].detach().cpu(), bsp_pred[1].detach().cpu()), dim = 0)
            bsp_truth_epoch = torch.cat((bsp_truth_epoch, bsp_lig.cpu(), bsp_rec.cpu()), dim = 0)
        
        if args['cp']:
            cp_loss = cp_loss_func(contact_info[0].squeeze(), contact_info[1].squeeze())
            cp_loss_epoch += cp_loss.item()
            loss += cp_loss
            AP_cp = average_precision(contact_info[0].squeeze().detach().cpu(), contact_info[1].squeeze().detach().cpu(), task="binary")
            AUC_cp = auroc(contact_info[0].squeeze().detach().cpu(), contact_info[1].squeeze().detach().cpu())
            APs_cp.append(AP_cp)
            AUCs_cp.append(AUC_cp)

        loss.backward()
        optimizer.step()
        
    print('bsp_loss:', bsp_loss_epoch / (id + 1), \
        '  mdn_loss:', mdn_loss_epoch / (id + 1), \
            '  cd_loss:', cd_loss_epoch / (id + 1), \
            ' cp_loss:', cp_loss_epoch / (id + 1))
    log_file.write('bsp_loss: ' + str(bsp_loss_epoch / (id + 1)) + \
        '  mdn_loss:' + str(mdn_loss_epoch / (id + 1)) +  \
            '  cd_loss:' + str(cd_loss_epoch / (id + 1)) + \
            ' cp_loss:' +  str(cp_loss_epoch / (id + 1)) + '\n')
    
    if args['bsp']:
        AP = average_precision(bsp_pred_epoch.squeeze(), bsp_truth_epoch, task="binary")
        AUC = auroc(bsp_pred_epoch.squeeze(), bsp_truth_epoch)
        print('Train bsp AP:', AP, 'Train bsp AUC:', AUC)
        log_file.write('Train bsp AP:' + str(AP) + 'Train bsp AUC:' + str(AUC) + '\n')
    
    if args['cp']:
        AP = sum(APs_cp) / len(APs_cp)
        AUC = sum(AUCs_cp) / len(AUCs_cp)
        print('Train cp AP:', AP, 'Train cp AUC:', AUC)
        log_file.write('Train cp AP:' + str(AP) + 'Train cp AUC:' + str(AUC) + '\n')
        
            
def main():
    
    np.random.seed(args['random_seed'])
    torch.cuda.manual_seed_all(args['random_seed'])
    torch.manual_seed(args['random_seed'])
    random.seed(args['random_seed'])
    
    log_file = open(args['log_file'], 'a')
    log_file.write(str(args) + "\n")
    

    model = energy_model(args).to(device)
    sample_model = model_sampler(args).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-5)
    bsp_loss_func = FocalLoss()
    cp_loss_func = FocalLoss(args['cp_alpha'])    
    
    train_dataset = DockingDataset(args, args['dataset'], reload_mode='train', load_from_cache=False, raw_data_path=args['data_path'])
    train_dataloader = DataLoader(train_dataset, batch_size=args['bs'], shuffle=True, collate_fn=docking_collate_fn)
    
    val_dataset =  DockingDataset(args, args['dataset'], reload_mode='val', load_from_cache=False, raw_data_path=args['data_path'])
    val_dataloader = DataLoader(val_dataset, batch_size=args['bs'], shuffle=False, collate_fn=docking_collate_fn)
     
    last_metric_1, last_metric_2 = 0, 1e9
    for epoch_id in range(args['n_epochs']):
        
        train_one_epoch(train_dataloader, model, sample_model, bsp_loss_func, cp_loss_func, optimizer, args, log_file)
        P, metric_val = val(val_dataloader, model, sample_model, args, log_file)
        
        ## for cp / bsp prediction
        if args['bsp'] or args['cp']:
            torch.save(model.state_dict(), args['save_path_prediction'] + 'contact_prediction_{}.pt'.format(epoch_id))
            if P > last_metric_1:
                last_metric_1 = P
                torch.save(model.state_dict(), args['save_path_prediction'] + 'contact_prediction_best.pt')
                print('Save binding interface prediction model')
        # for energy score function
        if args['ll_train'] or args['ebm_train']:
            torch.save(model.state_dict(), args['save_path_prediction'] + 'nenrgy_{}.pt'.format(epoch_id))
            if last_metric_2 > metric_val:
                last_metric_2 = metric_val
                torch.save(model.state_dict(), args['save_path_energy'] + 'energy_best.pt')
                print('Save energy model')
                        
    
if __name__ == '__main__':
 
    main()