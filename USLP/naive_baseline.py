import pandas as pd
import numpy as np
import math


import pickle
import h5py

from tqdm import tqdm
import re
import pdb
import logging
import os
import textdistance



from approach_utils import (
    set_logger, triple_uri_to_idx,
    naivebaseline_score,
    get_metrics
)

if '__main__' == __name__:

    # Write logs to checkpoint and console
    save_path = 'Results/<save_path>'
    data_path = '<data_path>'
    label_column_name = 'nameEn' # label column of naive baseline
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
    set_logger(log_file = os.path.join(save_path, 'test_log.log'))



    ##
    # read data
    ##
    entities_data = pd.read_csv(os.path.join(data_path,'final_approach_data_with_geohash_and_cluster_center.csv'))
    logging.info(f'entities_data.shape {entities_data.shape}')


    # create entity uri_idx dict from final approach entities data above
    entity_uri_idx_dict = dict()
    for i in range(len(entities_data['entity_uri'])):
        entity_uri_idx_dict[entities_data['entity_uri'][i]] = i


    # read test set triples
    # use relation_idx dictionary here and not relation 'name' idx dictionary
    test_triples = pd.read_csv(os.path.join(data_path,'test.txt'), sep = '\t', names = ['head', 'relation', 'tail'])
    test_triples = triple_uri_to_idx(test_triples, entity_uri_idx_dict, keep_rel_uris = True)
    logging.info(f'test_triples {len(test_triples)}')

    # candidate_idx for scoring
    candidate_idxs = np.arange(0, len(entities_data))


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
            head_idx, rel_uri, true_tail_idx = sample[0].astype(int), sample[1], sample[2].astype(int)

            candidate_idxs = np.arange(0, len(entities_data))
        
            # # compute score
            labels = entities_data[label_column_name].astype(str)
            tail_scores = naivebaseline_score(
                head_idx,
                labels
            )

            # compute metrics for a single sample & store in metrics_list
            metrics_list.append(
                get_metrics(
                    tail_scores,
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








