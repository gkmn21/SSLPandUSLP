import pandas as pd
import numpy as np
import math

import pickle
import h5py

from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm
import re
import pdb
import logging
import os
import argparse

from approach_utils import (
    parse_args, set_logger, triple_uri_to_idx,
    space1_geohash_score, space2_score, space3_score, space4_score,
    get_metrics
)

if '__main__' == __name__:

    data_dir = '../Datasets'
    args = parse_args()
    save_path = args.save_path
    data_path = os.path.join(data_dir, args.data_path)
    # score flags for ablation study
    with_score1 = args.with_score1
    with_score2 = args.with_score2
    with_score3 = args.with_score3
    with_SBERT_score = args.with_SBERT_score
    print(args)


    # Write logs to checkpoint and console
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
    set_logger(log_file = os.path.join(save_path, 'test_log.log'))



    ##
    # read data
    ##
    entities_data = pd.read_csv(os.path.join(data_path,'final_approach_data.csv'))
    logging.info(f'entities_data.shape {entities_data.shape}')

    #  Read Space 1 embeddings
    with h5py.File(os.path.join(data_path,'precision1_similarity_matrix_file.h5'), 'r') as hf:
        precision1_similarity_matrix = hf['precision1_similarity_matrix'][:]
    precision1_similarity_matrix = precision1_similarity_matrix.astype(dtype = np.float16, copy = False)
    logging.info(f'precision1_similarity_matrix.shape {precision1_similarity_matrix.shape}')

    with h5py.File(os.path.join(data_path,'precision3_similarity_matrix_file.h5'), 'r') as hf:
        precision3_similarity_matrix = hf['precision3_similarity_matrix'][:]
    precision3_similarity_matrix = precision3_similarity_matrix.astype(dtype = np.float16, copy = False)
    logging.info(f'precision3_similarity_matrix.shape {precision3_similarity_matrix.shape}')

    with h5py.File(os.path.join(data_path,'precision4_similarity_matrix_file.h5'), 'r') as hf:
        precision4_similarity_matrix = hf['precision4_similarity_matrix'][:]
    precision4_similarity_matrix = precision4_similarity_matrix.astype(dtype = np.float16, copy = False)
    logging.info(f'precision4_similarity_matrix.shape {precision4_similarity_matrix.shape}')
    
    precision1_cluster_center_geohashes = np.unique(entities_data['1_precision_cluster_center'])
    precision3_cluster_center_geohashes = np.unique(entities_data['3_precision_cluster_center'])
    precision4_cluster_center_geohashes = np.unique(entities_data['4_precision_cluster_center'])


    # Space 2 data
    ##  embeddings
    with h5py.File(os.path.join(data_path, 'mean_label_nameEn_embeddings_file.h5'), 'r') as hf:
        mean_label_nameEn_embeddings = hf['mean_label_nameEn_embeddings'][:]
    logging.info(f'mean_label_nameEn_embeddings.shape {mean_label_nameEn_embeddings.shape}')

    with h5py.File(os.path.join(data_path,'test_set_literal_value_embeddings_file.h5'), 'r') as hf:
        test_set_literal_value_embeddings = hf['test_set_literal_value_embeddings'][:]
    logging.info(f'test_set_literal_value_embeddings.shape {test_set_literal_value_embeddings.shape}')

    # Space 3 data
    ## type and relation embeddings
    with h5py.File(os.path.join(data_path,'type_and_relation_embeddings_file.h5'), 'r') as hf:
        type_and_relation_embeddings = hf['type_embeddings'][:]
    logging.info(f'type_and_relation_embeddings.shape {type_and_relation_embeddings.shape}')


    with open(os.path.join(data_path,'type_to_idx_dict.dict'), 'rb') as handle:
        type_to_idx_dict = pickle.load(handle)
    logging.info(len(type_to_idx_dict.keys()))

    with open(os.path.join(data_path,'relation_name_idx_dict.dict'), 'rb') as handle:
        relation_name_idx_dict = pickle.load(handle)
    logging.info(len(relation_name_idx_dict.keys()))

    # Dynamic embeddings
    ## entity SBERT  embeddings
    if with_SBERT_score:
        with h5py.File(os.path.join(data_path,'entity_tag_value_SBERT_embeddings_file.h5'), 'r') as hf:
            entity_tag_value_SBERT_embeddings = hf['entity_tag_value_SBERT_embeddings'][:]
        logging.info(f'entity_tag_value_SBERT_embeddings.shape {entity_tag_value_SBERT_embeddings.shape}')





    ##
    # generate data
    ##
    _entities_data_type = entities_data['type'].fillna('<UNK>')
    logging.info(f'_entities_data_type.shape {_entities_data_type.shape}')

    # create entity uri_idx dict from final approach entities data above
    entity_uri_idx_dict = dict()
    for i in range(len(entities_data['entity_uri'])):
        entity_uri_idx_dict[entities_data['entity_uri'][i]] = i

    
    # define no. of characters of geohash to be compared for filtering candidates
    geohash_precision_dictionary = {
        'wkgs:isIn': 1,
        'wkgs:addrPlace': 1,
        'wkgs:isInContinent': 1,
        'wkgs:country': 1,
        'wkgs:isInCountry': 1,
        'wkgs:addrCountry': 1,
        'wkgs:capitalCity': 1,
        'wkgs:addrState': 3,
        'wkgs:addrDistrict': 3,
        'wkgs:addrProvince': 3,
        'wkgs:isInCounty': 3,
        'wkgs:addrSubdistrict': 4,
        'wkgs:addrSuburb': 4,
        'wkgs:addrHamlet': 4
    }

    # relation name and idx dictionary
    relation_idx = {
        'wkgs:isIn': 0,
        'wkgs:addrPlace': 1,
        'wkgs:isInContinent': 2,
        'wkgs:country': 3,
        'wkgs:isInCountry': 4,
        'wkgs:addrCountry': 5,
        'wkgs:capitalCity': 6,
        'wkgs:addrState': 7,
        'wkgs:addrDistrict': 8,
        'wkgs:addrProvince': 9,
        'wkgs:isInCounty': 10,
        'wkgs:addrSubdistrict': 11,
        'wkgs:addrSuburb': 12,
        'wkgs:addrHamlet': 13
    }

    # read test set triples
    # !!! CAUTION - use relation_idx dictionary here and not relation 'name' idx dictionary
    test_triples = pd.read_csv(os.path.join(data_path,'test.txt'), sep = '\t', names = ['head', 'relation', 'tail'])
    test_triples = triple_uri_to_idx(test_triples, entity_uri_idx_dict, relation_idx, keep_rel_uris = True)
    logging.info(f'test_triples {len(test_triples)}')

    # precompute and store all_candidates tail_type_emb
    candidate_idxs = np.arange(0, len(entities_data))
    candidates_type_array = _entities_data_type[candidate_idxs].to_numpy()
    type_emb_idxs = [type_to_idx_dict[tail_type_uri] for tail_type_uri in candidates_type_array]
    all_candidates_type_embeddings = type_and_relation_embeddings[type_emb_idxs]
    logging.info(f'all_candidates_type_embeddings.shape {all_candidates_type_embeddings.shape}')

    ###
    ## Score and compute metrics
    ###
    ###

    
    # split test triples into batches
    n_triples = len(test_triples)
    batch_size = 64
    n_batches = math.ceil(n_triples / batch_size)

    metrics_list = []
    for i in tqdm(range(n_batches)):
        start = i * batch_size
        end =  (i+1) * batch_size if ((i+1) * batch_size) <= n_triples else n_triples
        batch_triples = np.array(test_triples[start: end])

        # for each sample score all tails
        for j, sample in enumerate(batch_triples):
            scores_list = []
            head_idx, rel_uri, true_tail_idx = sample[0].astype(int), sample[1], sample[2].astype(int)

            candidate_idxs = np.arange(0, len(entities_data))

            ## compute score in three spaces for h, r, t_batch)

            if with_score1:
                score1 = space1_geohash_score(
                    head_idx,
                    candidate_idxs,
                    rel_uri,
                    entities_data_with_geohash = entities_data,
                    geohash_precision_dictionary = geohash_precision_dictionary,
                    precision1_similarity_matrix = precision1_similarity_matrix,
                    precision3_similarity_matrix = precision3_similarity_matrix,
                    precision4_similarity_matrix = precision4_similarity_matrix,
                    precision1_cluster_center_geohashes = precision1_cluster_center_geohashes,
                    precision3_cluster_center_geohashes = precision3_cluster_center_geohashes,
                    precision4_cluster_center_geohashes = precision4_cluster_center_geohashes
                )
                scores_list.append(score1.reshape(1, -1))

            if with_score2:
                # score 2 - cosine sim between literal value & labelNameEn emb.
                score2 = space2_score(
                    literal_embedding = test_set_literal_value_embeddings[start + j],
                    all_candidates_label_nameEn_embeddings = mean_label_nameEn_embeddings
                )
                scores_list.append(score2)

            if with_score3:
                # precomputing and assuming all entities are used here
                score3 = space3_score(
                    relation_uri = rel_uri,
                    all_candidates_type_embeddings = all_candidates_type_embeddings,
                    type_and_relation_embeddings = type_and_relation_embeddings,
                    relation_name_idx_dict = relation_name_idx_dict
                )
                scores_list.append(score3)


            # score against embeddings of all entities
            if with_SBERT_score:
                score4 = space4_score(
                    head_idx = head_idx,
                    entity_tag_value_SBERT_embeddings = entity_tag_value_SBERT_embeddings
                )
                scores_list.append(score4) 


            # cumulative score from all spaces
            if 1 < len(scores_list):
                tail_scores = np.sum(scores_list, axis = 0)
            else:
                # if there is only one score with shape (1, n_entities)
                tail_scores = scores_list[0]

            # compute metrics for a single sample & store in metrics_list
            metrics_list.append(
                get_metrics(
                    tail_scores[0], # since tail_scores has shape (1, n_entities) extract tail_scores[0]
                    (head_idx, rel_uri, true_tail_idx)
                )
            )

    
    # aggregate metrics over all samples in all batches
    aggregated_metrics = {}
    for metric in ['rank', 'mean_reciprocal_rank', 'hits@1', 'hits@3', 'hits@5', 'hits@10']:
        aggregated_metrics[metric] = sum([item[metric] for item in metrics_list])/len(metrics_list)
    
    print(aggregated_metrics)
    for key in ['rank', 'mean_reciprocal_rank', 'hits@1', 'hits@3', 'hits@5', 'hits@10']:
        logging.info(f'{key}: {aggregated_metrics[key]}')

    # save metrics for each sample in df
    metrics_df = pd.DataFrame.from_dict(metrics_list)
    metrics_df.to_csv(os.path.join(save_path, 'metrics_df.zip'), index = False)








