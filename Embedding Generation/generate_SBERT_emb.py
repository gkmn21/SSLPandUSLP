import csv
import torch

import pandas as pd
import numpy as np

from tqdm import tqdm
import time

import sys
import pickle


import h5py
import pdb

from sentence_transformers import SentenceTransformer


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model = SentenceTransformer('all-mpnet-base-v2')
    

    # read entity_idx
    with open('entity_idx.pickle', 'rb') as handle:
        entity_idx = pickle.load(handle)

    # read literal properties files
    literal_triples = pd.read_csv(
        'KeyValues_of_final_approach_entities.txt',
        sep = '\t',
        lineterminator='\n',
        names = ['subject', 'predicate', 'object']
    )
    print(f'literal_triples.shape {literal_triples.shape}')

    # concatenate predicate and object column for each row
    literal_triples['predicate_object'] = literal_triples['predicate'].str.cat(literal_triples['object'], sep = ' ')
    literal_triples['predicate_object'] = literal_triples['predicate_object'].astype(str)
    literal_triples['predicate_object'] = literal_triples['predicate_object'].str.lower()

    # group triples by subject uri and create list of 'predicate_object' values
    literal_triples_modified = literal_triples.groupby('subject')['predicate_object'].apply(list).reset_index()
    

    # create a column to store sentence and index
    literal_triples_modified['sentence'] = literal_triples_modified['predicate_object'].apply(lambda x: ' '.join(x))
    literal_triples_modified['uri_idx'] = literal_triples_modified['subject'].apply(lambda uri: entity_idx[uri])


    sentence_embeddings = model.encode(
        sentences = literal_triples_modified['sentence'],
        show_progress_bar = True,
        device = device,
        convert_to_numpy = True
    )
    print(sentence_embeddings.shape)

    # store embedding in corresponding place in numpy array
    n_entities = len(entity_idx)
    emb_dim = 768 # SBERT emb dim.
    entity_tag_value_SBERT_embeddings = np.zeros((n_entities, emb_dim))
    for idx in tqdm(range(sentence_embeddings.shape[0])):
        entity_idx = literal_triples_modified['uri_idx'][idx]
        entity_tag_value_SBERT_embeddings[entity_idx] = sentence_embeddings[idx]

    # sanity check
    print(f'{np.count_nonzero(entity_tag_value_SBERT_embeddings)}')
    pdb.set_trace()
    # save embedding array
    with h5py.File('entity_tag_value_SBERT_embeddings_file.h5', 'w') as hf:
        hf.create_dataset('entity_tag_value_SBERT_embeddings',  data = entity_tag_value_SBERT_embeddings, compression = 'gzip')



