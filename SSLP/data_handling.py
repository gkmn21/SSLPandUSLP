import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import os
import json
import pickle
import h5py
import pdb


from utilities import negative_tail_sampler, triple_uri_to_idx
from Constants import BatchType, ModeType, relation_uri_idx_dict

###
# Dataloaders and Datasets
###
# class to hold data
class DataManager:

    def __init__(
        self,
        device,
        data_folder,
        test_on_slice = None,
        use_key_value_emb = False,
        use_relation_emb = False,
        use_hierarchy_info = False
    ):
        '''
        data_folder: path to folder containing triples and embedding files
        use_key_value_emb: `True` if embeddings of heterogeneous properties should be part of input
        use_relation_emb: `True` if embeddings of relation name embeddings should be used
        '''
        self.read_data(device, data_folder, test_on_slice, use_key_value_emb, use_relation_emb, use_hierarchy_info)


    def read_data(self, device, data_folder, test_on_slice, use_key_value_emb, use_relation_emb, use_hierarchy_info):
        
        relation_idx_uri_dict = dict((val, key) for key, val in relation_uri_idx_dict.items())
        self.relation_dict = relation_uri_idx_dict

        # r_candidates_dict for negative sampling
        with open(os.path.join(data_folder, 'r_candidates_dict.pickle'), 'rb') as handle:
            self.relation_candidates_dict = pickle.load(handle)
        
        with open(os.path.join(data_folder, 'entity_idx.pickle'), 'rb') as handle:
            self.entity_dict = pickle.load(handle)
        entity_idx_uri_dict = dict((val, key) for key, val in self.entity_dict.items())

        # entity_data
        self.data = pd.read_csv(os.path.join(data_folder, 'final_approach_data.csv'))
        print(f'data.shape: {self.data.shape}')

        # read input embeddings
        with h5py.File(os.path.join(data_folder, 'input_embeddings_file.h5'), 'r') as hf:
            input_embeddings = hf['input_embeddings'][:]
        self.input_embeddings = torch.tensor(input_embeddings, dtype = torch.float32)
        print(f'input_embeddings: {self.input_embeddings.shape}')

        if use_key_value_emb:
            # read key-value embeddings
            with h5py.File(os.path.join(data_folder, 'entity_tag_value_SBERT_embeddings_file.h5'), 'r') as hf:
                entity_tag_value_embedding = hf['entity_tag_value_SBERT_embeddings'][:]
            self.entity_tag_value_embedding = torch.tensor(entity_tag_value_embedding, dtype = torch.float32)
            print(f'entity_tag_value_embedding: {self.entity_tag_value_embedding.shape}')

        if use_relation_emb:
            # read input embeddings
            with h5py.File(os.path.join(data_folder, 'relation_names_FT_embeddings_file.h5'), 'r') as hf:
                rel_embeddings = hf['relation_names_FT_embeddings'][:]
            self.rel_embeddings = torch.tensor(rel_embeddings, dtype = torch.float32)
            print(f'rel_embeddings: {self.rel_embeddings.shape}')
            # pdb.set_trace()

        if use_hierarchy_info:
            # read entity_hierarchy level list
            with h5py.File(os.path.join(data_folder, 'entity_hierarchy_level_file.h5'), 'r') as hf:
                entity_hierarchy_level = hf['entity_hierarchy_level_embeddings'][:]
            self.entity_hierarchy_level = torch.tensor(entity_hierarchy_level).to(device)
            print(f'entity_hierarchy_level: {self.entity_hierarchy_level.shape}')
            
            # read r_direction list
            with h5py.File(os.path.join(data_folder, 'r_direction_file.h5'), 'r') as hf:
                r_direction = hf['r_direction_embeddings'][:]
            self.r_direction = torch.tensor(r_direction).to(device)
            print(f'r_direction: {self.r_direction.shape}')


        # train, val, test triples
        self.train_data = triple_uri_to_idx(os.path.join(data_folder, 'train.txt'), self.entity_dict, self.relation_dict)
        self.valid_data = triple_uri_to_idx(os.path.join(data_folder, 'valid.txt'), self.entity_dict, self.relation_dict)
        self.test_data = triple_uri_to_idx(os.path.join(data_folder, 'test.txt'), self.entity_dict, self.relation_dict)

        if test_on_slice:
            self.train_data = self.train_data[:test_on_slice]
            self.valid_data = self.valid_data[:test_on_slice]
            self.test_data = self.test_data[:test_on_slice]

        print(f'train_triples: {len(self.train_data)}')
        print(f'valid_triples: {len(self.valid_data)}')
        print(f'test_triples: {len(self.test_data)}')







class TrainDataset(Dataset):
        
    def __init__(self, data_manager, with_type_sampler, neg_size, batch_type):
        '''
        Training dataset
        - data_manager
        - with_type_sampler
        - neg_size
        '''

        self.triples = data_manager.train_data
        self.len = len(self.triples)
        self.num_entity = len(data_manager.entity_dict)
        self.num_relations = len(data_manager.relation_dict)
        self.neg_size = neg_size
        self.with_type_sampler = with_type_sampler
        self.relation_candidates_dict = data_manager.relation_candidates_dict
        self.batch_type = batch_type
        self.hr_map, self.tr_map, self.hr_freq, self.tr_freq = self.two_tuple_count()

    def __len__(self):
        return self.len

    
    def __getitem__(self, idx: int):
        """
        Returns a positive sample and `self.neg_size` negative samples.
        """
        pos_triple = self.triples[idx]
        head, rel, tail = pos_triple

        subsampling_weight = self.hr_freq[(head, rel)] + self.tr_freq[(tail, rel)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))


        if self.with_type_sampler:
            neg_triples = negative_tail_sampler(
                head,
                rel,
                tail,
                self.relation_candidates_dict,
                self.num_entity,
                n_samples = self.neg_size
            )
        
        else:

            # using random sampling
            neg_triples = []
            neg_size = 0

            while neg_size < self.neg_size:
                neg_triples_tmp = np.random.randint(self.num_entity, size=self.neg_size * 2)
                if self.batch_type == BatchType.TAIL_BATCH:
                    mask = np.in1d(
                        neg_triples_tmp,
                        self.hr_map[(head, rel)],
                        assume_unique=True,
                        invert=True
                    )
                else:
                    raise ValueError('Invalid BatchType: {}'.format(self.batch_type))

                neg_triples_tmp = neg_triples_tmp[mask]
                neg_triples.append(neg_triples_tmp)
                neg_size += neg_triples_tmp.size

            neg_triples = np.concatenate(neg_triples)[:self.neg_size]
            neg_triples = torch.from_numpy(neg_triples)
        
        pos_triple = torch.LongTensor(pos_triple)

        return pos_triple, neg_triples, subsampling_weight, self.batch_type

    
    def two_tuple_count(self):
        """
        Return two dict:
        dict({(h, r): [t1, t2, ...]}),
        dict({(t, r): [h1, h2, ...]}),
        """
        hr_map = {}
        hr_freq = {}
        tr_map = {}
        tr_freq = {}

        init_cnt = 3
        for head, rel, tail in self.triples:
            if (head, rel) not in hr_map.keys():
                hr_map[(head, rel)] = set()

            if (tail, rel) not in tr_map.keys():
                tr_map[(tail, rel)] = set()

            if (head, rel) not in hr_freq.keys():
                hr_freq[(head, rel)] = init_cnt

            if (tail, rel) not in tr_freq.keys():
                tr_freq[(tail, rel)] = init_cnt

            hr_map[(head, rel)].add(tail)
            tr_map[(tail, rel)].add(head)
            hr_freq[(head, rel)] += 1
            tr_freq[(tail, rel)] += 1

        for key in tr_map.keys():
            tr_map[key] = np.array(list(tr_map[key]))

        for key in hr_map.keys():
            hr_map[key] = np.array(list(hr_map[key]))

        return hr_map, tr_map, hr_freq, tr_freq

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim = 0)
        negative_sample = torch.stack([_[1] for _ in data], dim = 0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        batch_type = data[0][3]
        return positive_sample, negative_sample, subsample_weight, batch_type

class TestDataset(Dataset):

    def __init__(self, data_manager, mode, batch_type):
        
        self.triple_set = set(data_manager.train_data + data_manager.valid_data + data_manager.test_data)
        self.mode = mode
        if ModeType.VALIDATION == self.mode:
            self.triples = data_manager.valid_data
        elif ModeType.TEST == self.mode:
            self.triples = data_manager.test_data

        self.len = len(self.triples)
        self.num_entity = len(data_manager.entity_dict)
        self.num_relation = len(data_manager.relation_dict)
        self.batch_type = batch_type
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]

        # for filter_bias
        if BatchType.TAIL_BATCH == self.batch_type:
            tmp = [(0, rand_tail) if (h, r, rand_tail) not in self.triple_set
                    else (-1, t) for rand_tail in range(self.num_entity)]
            tmp[t] = (0, t)
            tmp = torch.LongTensor(tmp)
            filter_bias = tmp[:, 0].float()
            negative_sample = tmp[:, 1]
        positive_sample = torch.LongTensor((h, r, t))

        return positive_sample, negative_sample, filter_bias, self.batch_type

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim = 0)
        negative_sample = torch.stack([_[1] for _ in data], dim = 0)
        filter_bias = torch.stack([_[2] for _ in data], dim = 0)
        batch_type = data[0][3]

        return positive_sample, negative_sample, filter_bias, batch_type




