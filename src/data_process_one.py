import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os, random, dill, gzip, shutil, pickle
from scipy.spatial.transform import Rotation
from sklearn.neighbors import BallTree
from biopandas.pdb import PandasPdb
import torch
import math

def get_residues_one(receptor_name, ligand_name):

    df = PandasPdb().read_pdb(ligand_name).df['ATOM']
    df.rename(columns={'chain_id': 'chain', 'residue_number': 'residue', 'residue_name': 'resname',
                       'x_coord': 'x', 'y_coord': 'y', 'z_coord': 'z', 'element_symbol': 'element'}, inplace=True)
    residues0 = list(df.groupby(['chain', 'residue', 'resname']))  ## Not the same as sequence order !
    
    df = PandasPdb().read_pdb(receptor_name).df['ATOM']
    df.rename(columns={'chain_id': 'chain', 'residue_number': 'residue', 'residue_name': 'resname',
                       'x_coord': 'x', 'y_coord': 'y', 'z_coord': 'z', 'element_symbol': 'element'}, inplace=True)
    residues1 = list(df.groupby(['chain', 'residue', 'resname']))  ## Not the same as sequence order !
    
    return residues0, residues1, receptor_name

def zerocopy_from_numpy(x):
    return torch.from_numpy(x)

def UniformRotation_Translation(translation_interval):
    rotation = Rotation.random(num=1)
    rotation_matrix = rotation.as_matrix().squeeze()

    t = np.random.randn(1, 3)
    t = t / np.sqrt( np.sum(t * t))
    length = np.random.uniform(low=0, high=translation_interval)
    t = t * length
    return rotation_matrix.astype(np.float32), t.astype(np.float32)


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

    
def docking_collate_fn_inf(batch):
    
    pos_lig, pos_rec, atom_lig, atom_rec, batch_lig, batch_rec = [], [], [], [], [], []
    pos_lig_new = []

    for id, item in enumerate(batch):
        pos_lig.append(item['lig_pos'])
        pos_rec.append(item['rec_pos'])
        atom_lig.append(item['lig_atom'])
        atom_rec.append(item['rec_atom'])
        batch_lig.append(torch.full((item['lig_pos'].shape[0],),id))
        batch_rec.append(torch.full((item['rec_pos'].shape[0],),id))
        pos_lig_new.append(item['lig_new_pos'])
    
    pos_lig, pos_rec, atom_lig, atom_rec, batch_lig, batch_rec = torch.cat(pos_lig, dim=0), torch.cat(pos_rec, dim=0),\
                                            torch.cat(atom_lig, dim=0), torch.cat(atom_rec, dim=0),\
                                            torch.cat(batch_lig, dim=0), torch.cat(batch_rec, dim=0)
    
    return pos_lig, pos_rec, atom_lig, atom_rec, batch_lig, batch_rec, pos_lig_new

def rigid_transform_Kabsch_3D(A, B):
    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")


    # find mean column wise: 3 x 1
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # find rotation
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        SS = np.diag([1.,1.,-1.])
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(np.linalg.det(R) - 1) < 1e-5

    t = -R @ centroid_A + centroid_B
    return R, t

class Inference_Dataset(Dataset):
    
    def __init__(self, args):
        self.args = args
        self.if_swap = False
        
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
                
        residues0, residues1, dill_filename = get_residues_one(receptor_name=args['receptor_file_path'], ligand_name=args['ligand_file_path'])
                        
        bound_receptor_repres_nodes_loc_array, rec_atom = get_alphaC_loc_array(filter_residues(residues1))
        bound_ligand_repres_nodes_loc_array, lig_atom = get_alphaC_loc_array(filter_residues(residues0))


        assert bound_receptor_repres_nodes_loc_array is not None
        assert bound_ligand_repres_nodes_loc_array is not None
        
        data_item = {'rec_pos': bound_receptor_repres_nodes_loc_array,
                        'lig_pos': bound_ligand_repres_nodes_loc_array,
                        'filename': dill_filename,
                        'rec_atom': rec_atom,
                        'lig_atom': lig_atom}
        
        self.data.append(data_item)
             
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        

        rec_pos = self.data[idx]['rec_pos']
        lig_pos = self.data[idx]['lig_pos']
        rec_atom = self.data[idx]['rec_atom']
        lig_atom = self.data[idx]['lig_atom']  

        ligand_original_loc = lig_pos.copy()
        mean_to_remove = ligand_original_loc.mean(axis=0, keepdims=True)

        rot_T, rot_b = UniformRotation_Translation(translation_interval=self.args['translation_interval'])

        ligand_new_loc = (rot_T @ (ligand_original_loc - mean_to_remove).T).T + rot_b      
        
        data_item = {'rec_pos': zerocopy_from_numpy(rec_pos.astype(np.float32)),
                        'lig_pos': zerocopy_from_numpy(lig_pos.astype(np.float32)),
                        'lig_new_pos':zerocopy_from_numpy(ligand_new_loc.astype(np.float32)), 
                        'rec_atom': zerocopy_from_numpy(rec_atom.astype(np.int64)),
                        'lig_atom': zerocopy_from_numpy(lig_atom.astype(np.int64)),
                        'file_name': self.data[idx]['filename']
                        }
        
        return data_item