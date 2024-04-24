# DIPS-Het dataset generation
You can get the raw data and generate the training data by running:

1. sync with RCSB
```
rsync -rlpt -v -z --delete --port=33444 \
rsync.rcsb.org::ftp_data/biounit/coordinates/divided/ RCSB_pdb
```
2. obtain RCSB PDB metadata
   
```
python step1_query_rcsb_metadata.py
```

3. filter metadata

```   
python step2_filter_rcsb_metadata.py
```

4. generate ligand pdb and receptor pdb, split into train/val/test
   
```
python step3_pairs_to_pdb.py
```
    
