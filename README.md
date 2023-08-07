# Spatial Link Prediction with Spatial and Semantic Embeddings

This folder contains the scripts for the paper **Spatial Link Prediction with Spatial and Semantic Embeddings**.

## Folders
- `SSLP`, `USLP` - contain codes for the models SSLP and USLP respectively
- `Datasets` - contains TD1, TD2 and ID1 datasets.
- `Ground Truth Generation` - contains groundtruth generation scripts for creating TD1, TD2 and ID1 datasets.


## Dependencies
The `requirements.txt` has been provided for creating a conda environment with all dependencies. Dependencies include:
- Python 3.10.9
- PyTorch 2.0.0
- `fasttext`
- `sparqlwrapper`
- `pygeohash`
- `haversine`
- `h5py`
- `pykeen`


## SSLP
1. Run script `fetch_entities_info.py` to fetch entities data for all unique entities in training triples from WorldKG SPARQL endpoint, prepare entitiy idx, etc. 
```
python3 fetch_entities_info.py --data_folder 'Datasets/<dataset_name>' --data_filename 'train.txt' \
--entities_data_filename 'final_approach_data.csv' \
--hetero_prop_filename 'KeyValues_of_final_approach_entities.txt'
```

2. Run script `generate_embeddings.py` to generate spatial and semantic embeddings.
```
python3 generate_embeddings.py --data_folder 'Datasets/<dataset_name>' \
--gen_labelNameEn_emb --label_nameEn_merge 'sum' --gen_location_emb --gen_type_emb --gen_relationName_emb \
--entities_data_filename 'final_approach_data.csv' --gen_hierarchy_data \
--gen_dynamic_emb --hetero_prop_filename 'KeyValues_of_final_approach_entities.txt'
```

3. Navigate to folder `SSLP` (`cd SSLP`) and run the following command to reproduce the results of SSLP.
```
python3 main.py --do_train --do_test --data_path 'Datasets/<dataset_name>' --score_f HAKE --with_type_sampler -n 256 -b 64 -g 12.0 -a 1.0 -lr 0.001 --max_steps 3000 --log_steps 50 -save '<save_path>' --test_batch_size 1 -mw 1.0 -pw 0.5 -hw 0.8
```

### SSLP Baselines
1. The `SSLP\baselines.py` contains the script to execute baselines TransE, RotatE, ComplEx, DistMult, ConvKB, CompGCN. The variable `execution_config` needs to be changed in the script to specify the model, data_folder and save_path for baselines.

2. Source code provided by the publications was used for baselines HAKE, LiteralE, Literal2Entity-DistMult.


## USLP
1. Run script `fetch_entities_info.py` for unique entities in test triples, with the following arguments.
```
python3 fetch_entities_info.py --data_folder 'Datasets/<dataset_name>' --data_filename 'test.txt' \
--entities_data_filename 'final_approach_data.csv' --gen_similarity_matrices \
--hetero_prop_filename 'KeyValues_of_final_approach_entities.txt'
```

2. Run script `generate_embeddings.py` with the below arguments to prepare data and embeddings for USLP. Change the `--data_folder` argument to the folder containing the dataset.
```
python3 generate_embeddings.py --data_folder 'Datasets/<dataset_name>' \
--gen_labelNameEn_emb --label_nameEn_merge 'mean' --gen_type_and_relation_emb \
--entities_data_filename 'final_approach_data.csv' \
--gen_dynamic_emb --hetero_prop_filename 'KeyValues_of_final_approach_entities.txt' \
--gen_spatialprop_emb
```

3. Navigate to folder `USLP` (`cd USLP`) and run the following command.
```
python3 USLP_main.py --save_path <save_path> --data_path '<dataset_name>' --with_score1 --with_score2 \
--with_score3
```

### Inductive Naive Baseline
1. Navigate to folder `USLP` and run the following command
```
python3 naive_baseline.py --save_path 'Results/exp' --data_path '<dataset_name>' --label_column_name 'nameEn'
```


## Ground Truth Generation
Steps to generate datasets:
1. The file `Ground Truth Generation\spo_triples_iter1` contains ground truth triples generated using WorldKG.
2. Run the file `Ground Truth Generation\TD1_dataset_generation.ipynb` to generate TD1 dataset using triples present in `spo_triples_iter1`.
3. The file `Ground Truth Generation\wikidata_matched_triples.zip` contains ground truth generated using WorldKG Wikidata identity links.
4. Run the file `Ground Truth Generation\ID1_TD2_dataset_generation.ipynb` which generates ID1 and TD2 dataset by merging triples present in `wkg_gt.zip` and wikidata ground truth `wkg_wikidata_gt.zip`.