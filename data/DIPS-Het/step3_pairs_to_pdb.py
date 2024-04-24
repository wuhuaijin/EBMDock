"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import os
import time
import gzip
import shutil
import argparse
from tqdm import tqdm
import multiprocessing
from subprocess import Popen, PIPE
from functools import partialmethod

def convert_to_pdb(data_root, out_root, pair_name):
    # IO
    pdb_id = pair_name[:pair_name.find('_')]
    assert len(pdb_id) == 4
    pdbgz_fpath = os.path.join(data_root, pair_name[1:3].lower(), pdb_id.lower()+'.pdb1.gz')
    assert os.path.isfile(pdbgz_fpath)
    out_dir = out_root
    
    # complex pdb
    complex_pdb_fpath = os.path.join(out_dir, pair_name+'.pdb')
    with gzip.open(pdbgz_fpath, 'rb') as f_in:
        with open(complex_pdb_fpath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # split ligand-receptor pair
    lig_chain_id = pair_name.split('_')[1]
    assert len(lig_chain_id) == 1
    lig_pdb_lines = []
    rec_pdb_lines = []
    with open(complex_pdb_fpath, 'r') as fin:
        f_read = fin.readlines()
    for line in f_read:
        if line[:4] == 'ATOM':
            chain_id = line[21]
            if chain_id == lig_chain_id:
                lig_pdb_lines.append(line.strip())
            else:
                rec_pdb_lines.append(line.strip())
    
    # skip under/oversized cases and some exceptions (e.g., 1FVM, 3RUM)
    if min(len(lig_pdb_lines), len(rec_pdb_lines)) < 100 or \
       max(len(lig_pdb_lines), len(rec_pdb_lines)) > 12000:
        return

    # write ligand pdb            
    lig_pdb_fpath = os.path.join(out_dir, pair_name + '_l_b.pdb')
    with open(lig_pdb_fpath, 'w') as f:
        for line in lig_pdb_lines:
            f.write(line + '\n')
    # write receptor pdb
    rec_pdb_fpath = os.path.join(out_dir, pair_name + '_r_b.pdb')
    with open(rec_pdb_fpath, 'w') as f:
        for line in rec_pdb_lines:
            f.write(line + '\n')
    
    # os.remove(complex_pdb_fpath)   
    
            
def split_train_val():
    train_pdb = set()
    with open('train_pairs.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            train_pdb.add(line.strip())
    
    val_pdb = set()
    with open('val_pairs.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            val_pdb.add(line.strip())
            
    test_pdb = set()
    with open('test_pairs.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            test_pdb.add(line.strip())
            
    f_val = open('val.txt', 'w')
    f_train = open('train.txt', 'w')
    f_test = open('test.txt', 'w')
            
    with open('valid_pairs.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            name = line.strip()[:4]
            if name in train_pdb:
                f_train.write(line)
            if name in val_pdb:
                f_val.write(line)
            if name in test_pdb:
                f_test.write(line)
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb2pqr-bin', type=str, default='')
    parser.add_argument('--serial', action='store_true')
    parser.add_argument('-j', type=int, default=4)
    parser.add_argument('--mute-tqdm', action='store_true')
    args = parser.parse_args()
    print(args)

    # optionally mute tqdm
    if args.mute_tqdm:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    # RCSB
    rcsb_pdb_dir = '/RCSB_pdb/'
    assert os.path.exists(rcsb_pdb_dir)
    rcsb_mesh_dir = './pairs/'
    if os.path.exists(rcsb_mesh_dir):
        shutil.rmtree(rcsb_mesh_dir)
    os.makedirs(rcsb_mesh_dir, exist_ok=False)
    with open('valid_pairs.txt', 'r') as f:
        hetero_pairs = [pair_name.strip('\n') for pair_name in f.readlines()]

    start = time.time()

    if not args.serial:
        pool = multiprocessing.Pool(processes=args.j)
        pool_args = [(rcsb_pdb_dir, rcsb_mesh_dir, pair_name) 
                     for pair_name in hetero_pairs]
        pool.starmap(convert_to_pdb, tqdm(pool_args), chunksize=10)
        pool.terminate()
        print('All processes successfully finished')
    else:
        for pair_name in tqdm(hetero_pairs):
            convert_to_pdb(rcsb_pdb_dir, rcsb_mesh_dir, pair_name)
    
    print(f'RCSB step3 elapsed time: {(time.time()-start):.1f}s\n')
    
    split_train_val()

