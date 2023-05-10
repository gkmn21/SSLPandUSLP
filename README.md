# Spatial Link Prediction with Spatial and Semantic Embeddings

This folder contains the scripts for the paper **Spatial Link Prediction with Spatial and Semantic Embeddings**.

## Folders
- `SSLP`, `USLP` - contain codes for the models SSLP and USLP respectively
- `Datasets` - contains zip file with TD2 and ID1 datasets
- `Ground Truth Generation` - contains groundtruth generation files for creating TD1, TD2 and ID1 datasets
- `Embedding Generation` - contains scripts to generate spatial and semantic embeddings and other input files used by SSLP and USLP models

## Dependencies
- Python 3.10.9
- PyTorch 2.0.0
- `pip install fasttext`
- `pip install git+https://github.com/RDFLib/sparqlwrapper`
- `pip install pygeohash`
- `pip install haversine`
- `pip install h5py`
- `pip install pykeen`


## Ground Truth Generation
Steps to generate datasets:
1. The file `Ground Truth Generation\spo_triples_iter1` contains ground truth triples generated using WorldKG
2. Run the file `Ground Truth Generation\TD1_dataset_generation.ipynb` to generate TD1 dataset using triples present in `spo_triples_iter1`
3. The file `Ground Truth Generation\wikidata_matched_triples.zip` contains ground truth generated using WorldKG Wikidata identity links
4. Run the file `Ground Truth Generation\ID1_TD2_dataset_generation.ipynb` which generates ID1 and TD2 dataset by merging triples present in `spo_triples_iter1.zip` and wikidata ground truth `wikidata_matched_triples.zip`
5. The file `Ground Truth Generation\ID1_stratified_test.txt` contains the test set of ID1 dataset created after stratified sampling.
6. Run the script `fetch_literal_triples_wkg.py` to fetch all datatype properties from WorldKG, then run `fetch_literal_triples_coordinates.py` is to extract spatial coordinates from WorldKG.

## Embedding Generation for SSLP and USLP
1. Run the file `Embedding Generation\spatial_semantic_emb_generation.py` and `Embedding Generation\generate_SBERT_emb.py` to generate spatial and semantic embeddings. Place the embeddings in the `data` folders of SSLP and USLP.
2. Run the ipynb file `Embedding Generation\USLP_emb_generation.py` to generate input embeddings for USLP

## SSLP
1. Place `train.txt`, `valid.txt`, `test.txt` files of datasets and embedding files in the folder `SSLP\data`.

2. Navigate to folder `SSLP` and run the following command to reproduce the results of SSLP
```
python3 main.py --do_train --do_test --data_path <path_to_data_folder> --score_f HAKE -n 256 -b 64 -g 12.0 -a 1.0 -lr 0.001 --max_steps 3000 --log_steps 50 -save '<save_path>' --test_batch_size 1 -mw 1.0 -pw 0.5 -hw 0.8
```
### SSLP Baselines
1. The `SSLP\baselines.py` contains the script to execute baselines TransE, RotatE, ComplEx, DistMult, ConvKB, CompGCN. The variable `execution_config` needs to be changed to specify the model, data_folder and save_path for baselines.
2. Source code provided by the publications was used for baselines HAKE, LiteralE, Literal2Entity-DistMult. The variable `execution_config` needs to be changed to specify the model, data_folder and save_path for baselines.

## USLP
1. Place `train.txt`, `valid.txt`, `test.txt` and embedding files in the folder `USLP\data`.
2. Run the script `naive_baseine.py` to run LS naive baseline.
3. Run the script `USLP_main.py` to run the USLP model. The path to the data folder and result folder must be updated in the script.




