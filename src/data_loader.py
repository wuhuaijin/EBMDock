import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os, random, dill, gzip, shutil, pickle
from scipy.spatial.transform import Rotation
from sklearn.neighbors import BallTree
from biopandas.pdb import PandasPdb
import torch

def get_residues_DIPS_Het(pdb_filename):
    
    receptor_name = pdb_filename + '_r_b.pdb'
    ligand_name = pdb_filename + '_l_b.pdb'
    if not os.path.exists(receptor_name) or not os.path.exists(ligand_name):
        return None, None, None
    df0 = PandasPdb().read_pdb(ligand_name).df['ATOM']
    df0.rename(columns={'chain_id': 'chain', 'residue_number': 'residue', 'residue_name': 'resname',
                       'x_coord': 'x', 'y_coord': 'y', 'z_coord': 'z', 'element_symbol': 'element'}, inplace=True)
    residues0 = list(df0.groupby(['chain', 'residue', 'resname']))  ## Not the same as sequence order !
    df1 = PandasPdb().read_pdb(receptor_name).df['ATOM']
    df1.rename(columns={'chain_id': 'chain', 'residue_number': 'residue', 'residue_name': 'resname',
                       'x_coord': 'x', 'y_coord': 'y', 'z_coord': 'z', 'element_symbol': 'element'}, inplace=True)
    residues1 = list(df1.groupby(['chain', 'residue', 'resname']))  ## Not the same as sequence order !
    return residues0, residues1, pdb_filename


def get_residues_db5(pdb_filename):
    receptor_name = pdb_filename + '_r_b.pdb'
    ligand_name = pdb_filename + '_l_b.pdb'
    df = PandasPdb().read_pdb(ligand_name).df['ATOM']
    df.rename(columns={'chain_id': 'chain', 'residue_number': 'residue', 'residue_name': 'resname',
                       'x_coord': 'x', 'y_coord': 'y', 'z_coord': 'z', 'element_symbol': 'element'}, inplace=True)
    residues0 = list(df.groupby(['chain', 'residue', 'resname']))  ## Not the same as sequence order !
    
    df = PandasPdb().read_pdb(receptor_name).df['ATOM']
    df.rename(columns={'chain_id': 'chain', 'residue_number': 'residue', 'residue_name': 'resname',
                       'x_coord': 'x', 'y_coord': 'y', 'z_coord': 'z', 'element_symbol': 'element'}, inplace=True)
    residues1 = list(df.groupby(['chain', 'residue', 'resname']))  ## Not the same as sequence order !
    
    return residues0, residues1, pdb_filename

def get_residues_xTrimo(pdb_filename):
    receptor_name = pdb_filename + '_antigen.pdb'
    ligand_name = pdb_filename + '_antibody.pdb'
    df = PandasPdb().read_pdb(ligand_name).df['ATOM']
    df.rename(columns={'chain_id': 'chain', 'residue_number': 'residue', 'residue_name': 'resname',
                       'x_coord': 'x', 'y_coord': 'y', 'z_coord': 'z', 'element_symbol': 'element'}, inplace=True)
    residues0 = list(df.groupby(['chain', 'residue', 'resname']))  ## Not the same as sequence order !
    
    df = PandasPdb().read_pdb(receptor_name).df['ATOM']
    df.rename(columns={'chain_id': 'chain', 'residue_number': 'residue', 'residue_name': 'resname',
                       'x_coord': 'x', 'y_coord': 'y', 'z_coord': 'z', 'element_symbol': 'element'}, inplace=True)
    residues1 = list(df.groupby(['chain', 'residue', 'resname']))  ## Not the same as sequence order !
    
    return residues0, residues1, pdb_filename

def UniformRotation_Translation(translation_interval):
    rotation = Rotation.random(num=1)
    rotation_matrix = rotation.as_matrix().squeeze()

    t = np.random.randn(1, 3)
    t = t / np.sqrt( np.sum(t * t))
    length = np.random.uniform(low=0, high=translation_interval)
    t = t * length
    return rotation_matrix.astype(np.float32), t.astype(np.float32)

def zerocopy_from_numpy(x):
    return torch.from_numpy(x)


dit = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
           'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
           'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',

           'HIP': 'H', 'HIE': 'H', 'TPO': 'T', 'HID': 'H', 'LEV': 'L', 'MEU': 'M', 'PTR': 'Y',
           'GLV': 'E', 'CYT': 'C', 'SEP': 'S', 'HIZ': 'H', 'CYM': 'C', 'GLM': 'E', 'ASQ': 'D',
           'TYS': 'Y', 'CYX': 'C', 'GLZ': 'G'}

def residue_type_one_hot_dips_not_one_hot(residue):

    rare_residues = {'HIP': 'H', 'HIE': 'H', 'TPO': 'T', 'HID': 'H', 'LEV': 'L', 'MEU': 'M', 'PTR': 'Y',
           'GLV': 'E', 'CYT': 'C', 'SEP': 'S', 'HIZ': 'H', 'CYM': 'C', 'GLM': 'E', 'ASQ': 'D',
           'TYS': 'Y', 'CYX': 'C', 'GLZ': 'G'}

    if residue in rare_residues.keys():
        print('Some rare residue: ', residue)

    indicator = {'Y': 0, 'R': 1, 'F': 2, 'G': 3, 'I': 4, 'V': 5,
                 'A': 6, 'W': 7, 'E': 8, 'H': 9, 'C': 10, 'N': 11,
                 'M': 12, 'D': 13, 'T': 14, 'S': 15, 'K': 16, 'L': 17, 'Q': 18, 'P': 19}
    res_name = residue
    if res_name not in dit.keys():
        return 20
    else:
        res_name = dit[res_name]
        return indicator[res_name]
    

def docking_collate_fn(batch):
    
    pos_lig, pos_rec, atom_lig, atom_rec, batch_lig, batch_rec = [], [], [], [], [], []
    bsp_lig, bsp_rec = [], []
    file_name = []
    pos_lig_new = []
    RT = []
    
    for id, item in enumerate(batch):
        pos_lig.append(item['lig_pos'])
        pos_rec.append(item['rec_pos'])
        pos_lig_new.append(item['lig_new_pos'])
        atom_lig.append(item['lig_atom'])
        atom_rec.append(item['rec_atom'])
        bsp_lig.append(item['bsp_lig'])
        bsp_rec.append(item['bsp_rec'])
        batch_lig.append(torch.full((item['lig_pos'].shape[0],),id))
        batch_rec.append(torch.full((item['rec_pos'].shape[0],),id))
        file_name.append(item['file_name'])
        RT.append(item['RT'])
        
    
    pos_lig, pos_rec, atom_lig, atom_rec, batch_lig, batch_rec, bsp_lig, bsp_rec = torch.cat(pos_lig, dim=0), torch.cat(pos_rec, dim=0),\
                                            torch.cat(atom_lig, dim=0), torch.cat(atom_rec, dim=0),\
                                            torch.cat(batch_lig, dim=0), torch.cat(batch_rec, dim=0),\
                                            torch.cat(bsp_lig, dim=0), torch.cat(bsp_rec, dim=0)
    return pos_lig, pos_rec, atom_lig, atom_rec, batch_lig, batch_rec, bsp_lig, bsp_rec, file_name, pos_lig_new, RT
        

class DockingDataset(Dataset):
    
    def __init__(self, args, data_set, reload_mode, raw_data_path, load_from_cache=False, inference=False):
        self.args = args
        self.if_swap = True
        self.dataset = data_set
        if reload_mode == 'test' or inference:
            self.if_swap = False
        
        if load_from_cache:                
            self.data = pickle.load(open(os.path.join(raw_data_path, reload_mode +'.pkl'), 'rb'))
        else:
            if self.dataset == 'db5':
                dill_filenames_list = []
                file_path = os.path.join(raw_data_path, reload_mode + '.txt')
                with open(file_path, 'r') as f:
                    for line in f.readlines():
                        dill_filenames_list.append(line.rstrip())
                        
                print('Num of pairs in ', reload_mode, ' = ', len(dill_filenames_list))
                file_path = os.path.join(raw_data_path, 'structures')
                input_residues_lists = [get_residues_db5(os.path.join(file_path, f)) for f in dill_filenames_list]
                 
            elif self.dataset == 'dipshet':
                dill_filenames_list = []
                with open(os.path.join(raw_data_path, reload_mode + '.txt'), 'r') as f:
                    for line in f.readlines():
                        dill_filenames_list.append(line.rstrip())
                
                print('Num of pairs in ', reload_mode, ' = ', len(dill_filenames_list))
                file_path = os.path.join(raw_data_path, 'pairs')
                input_residues_lists = [get_residues_DIPS_Het(os.path.join(file_path, f)) for f in dill_filenames_list]

                
            elif self.dataset == 'antibody-antigen':
                dill_filenames_list = []
                with open('../data/antibody-antigen/pairs.txt', 'r') as f:
                    for line in f.readlines():
                        dill_filenames_list.append(line.rstrip())
                input_residues_lists = [get_residues_xTrimo(f) for f in dill_filenames_list]

            print("Done reading data")
            
            self.data = []
            
            def get_alphaC_loc_array(bound_predic_clean_list):
                bound_alphaC_loc_clean_list = []
                seq_atom = []
                for residue in bound_predic_clean_list:
                    df = residue[1]
                    alphaCatom = df[df['atom_name'] == 'CA']
                    alphaC_loc = alphaCatom[['x', 'y', 'z']].to_numpy().squeeze().astype(np.float32)
                    assert alphaC_loc.shape == (3,), \
                        f"alphac loc shape problem, shape: {alphaC_loc.shape} residue {df} resid {df['residue']}"
                    bound_alphaC_loc_clean_list.append(alphaC_loc)
                    seq_atom.append(residue_type_one_hot_dips_not_one_hot(alphaCatom['resname'].tolist()[0]))
                    
                if len(bound_alphaC_loc_clean_list) <= 1:
                    return None, None
                return np.stack(bound_alphaC_loc_clean_list, axis=0), np.stack(seq_atom, axis=0)  # (N_res,3)
            
            def filter_residues(residues):
                residues_filtered = []
                for residue in residues:
                    df = residue[1]
                    Natom = df[df['atom_name'] == 'N']
                    alphaCatom = df[df['atom_name'] == 'CA']
                    Catom = df[df['atom_name'] == 'C']

                    if Natom.shape[0] == 1 and alphaCatom.shape[0] == 1 and Catom.shape[0] == 1:
                        residues_filtered.append(residue)
                return residues_filtered
                    
            for i in range(len(input_residues_lists)):
                residues0, residues1, dill_filename = input_residues_lists[i]
                if residues0 is None:
                    continue
                
                print(dill_filename)
                                
                bound_receptor_repres_nodes_loc_array, rec_atom = get_alphaC_loc_array(filter_residues(residues1))
                bound_ligand_repres_nodes_loc_array, lig_atom = get_alphaC_loc_array(filter_residues(residues0))
                if bound_receptor_repres_nodes_loc_array is None or bound_ligand_repres_nodes_loc_array is None:
                    print('Skipping this pair')
                    continue
                
                data_item = {'rec_pos': bound_receptor_repres_nodes_loc_array,
                             'lig_pos': bound_ligand_repres_nodes_loc_array,
                             'filename': dill_filename,
                             'rec_atom': rec_atom,
                             'lig_atom': lig_atom}
                self.data.append(data_item)
            
            write_file_path = os.path.join(raw_data_path, reload_mode +'.pkl')
            pickle.dump(self.data, open(write_file_path, 'wb'))
            
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        swap = False
        if self.if_swap:
            rnd = np.random.uniform(low=0.0, high=1.0)
            if rnd > 0.5:
                swap = True
        
        if swap:
            rec_pos = self.data[idx]['lig_pos']
            lig_pos = self.data[idx]['rec_pos']
            rec_atom = self.data[idx]['lig_atom']
            lig_atom = self.data[idx]['rec_atom']
        else:
            rec_pos = self.data[idx]['rec_pos']
            lig_pos = self.data[idx]['lig_pos']
            rec_atom = self.data[idx]['rec_atom']
            lig_atom = self.data[idx]['lig_atom']        
        
        rot_T, rot_b = UniformRotation_Translation(translation_interval=self.args['translation_interval'])

        
        ligand_original_loc = lig_pos.copy()
        mean_to_remove = ligand_original_loc.mean(axis=0, keepdims=True)

        ligand_new_loc = (rot_T @ (ligand_original_loc - mean_to_remove).T).T + rot_b
        
        '''
        TODO:  numerical precision of inverse
        '''
        rot_T_ = Rotation.from_matrix(rot_T)
        rot_T_inv = rot_T_.inv()
        rot_b_inv = mean_to_remove - (rot_T_inv.as_matrix().squeeze() @ rot_b.T).T
        rot_T_inv_euler = np.flipud(rot_T_inv.as_euler('xyz'))
        rot_b_inv_vec = np.squeeze(rot_b_inv)
          
          
        if self.args['bsp'] or self.args['bsgt4mdn']:
            lig_gt = lig_pos
            rec_gt = rec_pos
            bt1 = BallTree(lig_gt)
            dist1, _ = bt1.query(rec_gt, k=1)
            bsp_rec = np.zeros(rec_gt.shape[0], dtype = np.int64)
            bsp_rec[np.where(dist1 <= self.args['bsp_threshold'])[0]] = 1
            bt2 = BallTree(rec_gt)
            dist1, _ = bt2.query(lig_gt, k=1)
            bsp_lig = np.zeros(lig_gt.shape[0], dtype = np.int64)
            bsp_lig[np.where(dist1 <= self.args['bsp_threshold'])[0]] = 1
        else:
            bsp_lig = np.zeros(0)
            bsp_rec = np.zeros(0)
            
        data_item = {'rec_pos': zerocopy_from_numpy(rec_pos.astype(np.float32)),
                        'lig_pos': zerocopy_from_numpy(lig_pos.astype(np.float32)),
                        'lig_new_pos':zerocopy_from_numpy(ligand_new_loc.astype(np.float32)), 
                        'rec_atom': zerocopy_from_numpy(rec_atom.astype(np.int64)),
                        'lig_atom': zerocopy_from_numpy(lig_atom.astype(np.int64)),
                        'bsp_rec': zerocopy_from_numpy(bsp_rec.astype(np.int64)),
                        'bsp_lig': zerocopy_from_numpy(bsp_lig.astype(np.int64)),
                        'file_name': self.data[idx]['filename'],
                        'RT': [rot_T, rot_b]}
        return data_item
            

# if __name__ == '__main__':
    
#     args = {'translation_interval': 10.0, 'bsp': True, 'bsgt4mdn': False, 'bsp_threshold': 8.0}
    
#     # Protein = DIPS_Het(args, reload_mode='val', load_from_cache=False, raw_data_path='/root/dock/DIPS-het/datadill', split_data_path='/root/dock/DIPS-het')
#     # Protein_train = DockingDataset(args, '', reload_mode='train', load_from_cache=False, raw_data_path='/root/dock/DIPS-het/datadill', split_data_path='/root/dock/DIPS-het')