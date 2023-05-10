import pandas as pd
import numpy as np
import math

import pickle
import h5py

from sklearn.metrics.pairwise import cosine_similarity
import textdistance

from tqdm import tqdm
import re
import pdb
import logging

def set_logger(log_file):
    '''
    Write logs to checkpoint and console
    '''

    
    logging.basicConfig(
        format = '%(asctime)s %(levelname)-8s %(message)s',
        level = logging.INFO,
        datefmt = '%Y-%m-%d %H:%M:%S',
        filename = log_file,
        filemode = 'w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)




def triple_uri_to_idx(triples, entity_uri_idx_dict, relation_uri_idx_dict = None, keep_rel_uris = True):

    data_idxs = []
    for i in range(len(triples)):
        head, relation, tail = triples.iloc[i]['head'], triples.iloc[i]['relation'], triples.iloc[i]['tail']
        relation = relation if keep_rel_uris else relation_uri_idx_dict[relation]
        data_idxs.append((entity_uri_idx_dict[head], relation, entity_uri_idx_dict[tail]))

    return data_idxs

#####
## SCORE FUNCTIONS
#####

# Score function - Space 1
def head_tail_geohash_center_similarity(
    h_cluster_center,
    t_cluster_center_batch,
    precision_similarity_matrix,
    precision_cluster_center_geohashes):

    
    # convert candidates array to shape (batch_size, 1)
    t_cluster_center_batch = t_cluster_center_batch.reshape(-1, 1)
    h_cluster_center_sim_matrix_idx = np.where(precision_cluster_center_geohashes == h_cluster_center)[0][0]
    t_cluster_center_sim_matrix_idx = np.where(precision_cluster_center_geohashes == t_cluster_center_batch)[1]

    return precision_similarity_matrix[h_cluster_center_sim_matrix_idx][t_cluster_center_sim_matrix_idx]

def space1_geohash_score(
    head_idx,
    candidate_idx_batch,
    relation_uri,
    entities_data_with_geohash,
    geohash_precision_dictionary,
    precision1_similarity_matrix,
    precision3_similarity_matrix,
    precision4_similarity_matrix,
    precision1_cluster_center_geohashes,
    precision3_cluster_center_geohashes,
    precision4_cluster_center_geohashes):
    '''
    return an array (batch_size, ) of similarity scores for all candidates in the batch
    '''

    # calculate grid precision for clustering
    precision = geohash_precision_dictionary[relation_uri]

    if 1 == precision:
        h_cluster_center = entities_data_with_geohash.iloc[head_idx]['1_precision_cluster_center']
        t_cluster_center_batch = entities_data_with_geohash.iloc[candidate_idx_batch]['1_precision_cluster_center'].to_numpy()
        precision_similarity_matrix =  precision1_similarity_matrix
        precision_cluster_center_geohashes = precision1_cluster_center_geohashes
    elif 3 == precision:
        h_cluster_center = entities_data_with_geohash.iloc[head_idx]['3_precision_cluster_center']
        t_cluster_center_batch = entities_data_with_geohash.iloc[candidate_idx_batch]['3_precision_cluster_center'].to_numpy()
        precision_similarity_matrix =  precision3_similarity_matrix
        precision_cluster_center_geohashes = precision3_cluster_center_geohashes
    elif 4 == precision:
        h_cluster_center = entities_data_with_geohash.iloc[head_idx]['4_precision_cluster_center']
        t_cluster_center_batch = entities_data_with_geohash.iloc[candidate_idx_batch]['4_precision_cluster_center'].to_numpy()
        precision_similarity_matrix =  precision4_similarity_matrix
        precision_cluster_center_geohashes = precision4_cluster_center_geohashes
    
    return head_tail_geohash_center_similarity(
        h_cluster_center,
        t_cluster_center_batch,
        precision_similarity_matrix,
        precision_cluster_center_geohashes
    )

# Score Space 2
def space2_score(
    literal_embedding,
    all_candidates_label_nameEn_embeddings):

    return cosine_similarity(
        literal_embedding.reshape(1, -1),
        all_candidates_label_nameEn_embeddings
    )

# Score Space 3
def space3_score(
    relation_uri,
    all_candidates_type_embeddings,
    type_and_relation_embeddings,
    relation_name_idx_dict):
    '''
    relation_uri
    all_candidates_type_embeddings: (batch_size, type_emb_dim)
    type_and_relation_embeddings
    '''

    relation_emb_idx = relation_name_idx_dict[relation_uri]
    

    return cosine_similarity(
        type_and_relation_embeddings[relation_emb_idx].reshape(1, -1),
        all_candidates_type_embeddings
    )

# Score space 4
def space4_score(
    head_idx,
    entity_tag_value_SBERT_embeddings):
    
    # return util.cos_sim(entity_tag_value_SBERT_embeddings[head_idx],  entity_tag_value_SBERT_embeddings).numpy()
    return cosine_similarity(
        entity_tag_value_SBERT_embeddings[head_idx].reshape(1, -1),
        entity_tag_value_SBERT_embeddings
    )

# compute metrics
def get_metrics(all_candidates_scores, triple):

    # sort scores in descending order and get sorted indices
    sorted_score_descending_indices = all_candidates_scores.argsort()[::-1]

    true_tail_index = triple[2]
    ranking = (sorted_score_descending_indices == true_tail_index).nonzero()[0]
    ranking = ranking[0] + 1
    
    # pdb.set_trace()
    return {
        'rank': float(ranking),
        'mean_reciprocal_rank': 1.0 / ranking,
        'hits@1': 1.0 if ranking == 1 else 0.0,
        'hits@3': 1.0 if ranking <= 3 else 0.0,
        'hits@5': 1.0 if ranking <= 5 else 0.0,
        'hits@10': 1.0 if ranking <= 10 else 0.0,
        'triple': triple,
        'true_tail_score': all_candidates_scores[true_tail_index],
        'top_5_candidate_indices': sorted_score_descending_indices[:5]
    }

# naive baseline score
def naivebaseline_score(head_idx, labels):
    levenshteinSim = []
    for i in range(labels.shape[0]):
        levenshteinSim.append(
            textdistance.levenshtein.normalized_similarity(
                labels[head_idx], labels[i]
            )
        )
    return np.array(levenshteinSim)