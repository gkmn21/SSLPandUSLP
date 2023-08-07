'''
Script to generate embeddings for SSLP and USLP

Input: 
- train.txt/test.txt : triples file,
- entity_idx.pickle: entity index dictionary,
- final_approach_data.csv: entities data (label, nameEn, coordinates, etc.),
- KeyValues_of_final_approach_entities.txt: heterogeneous properties of all entities
'''

import glob
import sys
import os.path
import re
import csv
import pickle
import h5py
import pdb
from io import StringIO
import math
from tqdm import tqdm
import time
import argparse

import pandas as pd
import numpy as np
import fasttext 
import fasttext.util
import torch



from dataprep_utils import (
    get_ft_embedding, get_indices_from_uri_list,
    parse_args
)
from location_encoders import TheoryGridCellSpatialRelationEncoder
from Constants import relation_uri_idx_dict, relation_type_mapping, relation_rule_direction, type_hierarchy
from sentence_transformers import SentenceTransformer

if '__main__' == __name__:

    args = parse_args()
    data_folder = args.data_folder
    gen_labelNameEn_emb = args.gen_labelNameEn_emb
    gen_location_emb = args.gen_location_emb
    gen_type_emb = args.gen_type_emb
    gen_relationName_emb = args.gen_relationName_emb
    gen_type_and_relation_emb = args.gen_type_and_relation_emb # for USLP
    gen_hierarchy_data = args.gen_hierarchy_data # for SSLP hierarchy penalisation
    gen_spatialprop_emb = args.gen_spatialprop_emb
    gen_dynamic_emb = args.gen_dynamic_emb
    entities_data_filename = args.entities_data_filename
    hetero_prop_filename = args.hetero_prop_filename
    if gen_labelNameEn_emb:
        label_nameEn_merge = args.label_nameEn_merge
    print(args)

    if gen_labelNameEn_emb or gen_location_emb or gen_type_emb or gen_relationName_emb:
        # load fastText model
        fasttext.util.download_model('en', if_exists = 'ignore')  # English
        model = fasttext.load_model('cc.en.300.bin')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ##
    # Read and clean entities data
    ##
    train_entities_data = pd.read_csv(os.path.join(data_folder, entities_data_filename))
    print(f'train_entities_data.shape {train_entities_data.shape}')

    # fill unknown type with <UNK>
    train_entities_data['type'].fillna('<UNK>', inplace = True)

    # list of unique types
    unique_types, type_counts = np.unique(train_entities_data[['type']].values.flatten(), return_counts = True)
    print(f'unique_types.shape {unique_types.shape}')

    # create unique_types to idx dictionary
    type_to_idx_dict = dict()
    for i in range(len(unique_types)):
        type_to_idx_dict[unique_types[i]] = i
    
    # save type index as .dict file
    with open(os.path.join(data_folder, 'type_to_idx_dict.dict'), 'wb') as handle:
        pickle.dump(type_to_idx_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    ##
    # Generate data for hierarchy penalization
    ##
    if gen_hierarchy_data:
        
        # create list with relation direction, following same indexing as relation FT embeddings 
        r_direction = []
        for key in list(relation_rule_direction.keys()):
            r_direction.append(relation_rule_direction[key])
        print(f'len(r_direction) {len(r_direction)}')

        with h5py.File(os.path.join(data_folder, 'r_direction_file.h5'), 'w') as hf:
            hf.create_dataset('r_direction_embeddings',  data = r_direction, compression = 'gzip')

        # assign hierarchy levels to each rdf:type
        entity_hierarchy_level = []
        for _type in train_entities_data['type']:
            entity_hierarchy_level.append(type_hierarchy.get(_type, 6))
        print(f'len(entity_hierarchy_level) {len(entity_hierarchy_level)}')

        entity_hierarchy_level = torch.tensor(entity_hierarchy_level)
        with h5py.File(os.path.join(data_folder, 'entity_hierarchy_level_file.h5'), 'w') as hf:
            hf.create_dataset('entity_hierarchy_level_embeddings',  data = entity_hierarchy_level, compression = 'gzip')


    # remove 'wkgs:' prefix from type
    pattern = '^wkgs:'
    compiled_pattern = re.compile(pattern)
    train_entities_data['type'] = train_entities_data['type'].apply(lambda uri: re.sub(compiled_pattern, '', uri))


    # # create unique_types to idx dictionary
    # type_to_idx_dict = dict()
    # for i in range(len(unique_types)):
    #     type_to_idx_dict[unique_types[i]] = i
    
    # # save type index as .dict file
    # with open(os.path.join(data_folder, 'type_to_idx_dict.dict'), 'wb') as handle:
    #     pickle.dump(type_to_idx_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ##
    # fastText embedding of entity type
    ##
    if gen_type_emb:

        _property_name = 'type'
        property_numpy = train_entities_data[_property_name].to_numpy(copy = True)

        type_FT_embeddings = np.array([get_ft_embedding(x, model) for x in property_numpy], dtype = np.float64)

        # save as h5py
        with h5py.File(os.path.join(data_folder, 'entity_type_FT_embeddings_file.h5'), 'w') as hf:
            hf.create_dataset('entity_type_FT_embeddings',  data = type_FT_embeddings, compression = 'gzip')

    ##
    # fastText embedding of relation names
    ##
    if gen_relationName_emb:

        relation_names = list(relation_uri_idx_dict.keys())

        # remove 'wkgs:' prefix from relation_names
        pattern = '^wkgs:'
        compiled_pattern = re.compile(pattern)
        relation_names = [re.sub(compiled_pattern, '', uri) for uri in relation_names]
        print(f'Total relation_names: {len(relation_names)}')

        relation_names_FT_embeddings = np.array([get_ft_embedding(x, model) for x in relation_names], dtype = np.float64)
        print(f'relation_names_FT_embeddings.shape {relation_names_FT_embeddings.shape}')

        # save as h5py
        with h5py.File(os.path.join(data_folder, 'relation_names_FT_embeddings_file.h5'), 'w') as hf:
            hf.create_dataset('relation_names_FT_embeddings',  data = relation_names_FT_embeddings, compression = 'gzip')
        
        
    ##
    # creating single index and file of type relation embedding for USLP
    ##
    if gen_type_and_relation_emb:
        
        # update indices as continuation from rdf:type indices
        index_start = len(unique_types)
        relation_names = []
        relation_name_idx_dict = dict.fromkeys(relation_uri_idx_dict.values(), None)
        for key, val in relation_uri_idx_dict.items():
            relation_name_idx_dict[key] = val + index_start
            relation_names.append(key)
        
        # remove 'wkgs:' prefix from relation_names
        pattern = '^wkgs:'
        compiled_pattern = re.compile(pattern)
        relation_names = [re.sub(compiled_pattern, '', uri) for uri in relation_names]

        # merge relation names with type
        types_and_relation_names = np.concatenate((unique_types, np.array(relation_names)))
        print(f'types_and_relation_names.shape {types_and_relation_names.shape}')

        # generate embeddings stored at corresponding idx
        embeddings = np.array([get_ft_embedding(x, model) for x in types_and_relation_names], dtype = np.float64)

        # save as h5py
        with h5py.File(os.path.join(data_folder,'type_and_relation_embeddings_file.h5'), 'w') as hf:
            hf.create_dataset('type_embeddings',  data = embeddings, compression = 'gzip')

        # save type index as .dict file
        with open(os.path.join(data_folder, 'type_to_idx_dict.dict'), 'wb') as handle:
            pickle.dump(type_to_idx_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # save relation_name index as .dict file
        with open(os.path.join(data_folder, 'relation_name_idx_dict.dict'), 'wb') as handle:
            pickle.dump(relation_name_idx_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ##
    # fastText embedding of label nameEn
    ##
    if gen_labelNameEn_emb:

        # embedding of nameEn
        _property_name = 'nameEn'
        train_entities_data[_property_name].fillna('<UNK>', inplace = True)
        train_entities_data[pd.isna(train_entities_data[_property_name])]

        property_numpy = train_entities_data[_property_name].to_numpy(copy = True)

        nameEn_embeddings = np.array([get_ft_embedding(x, model) for x in property_numpy], dtype = np.float64)

        # save as h5py
        with h5py.File(os.path.join(data_folder, 'nameEn_embeddings_file.h5'), 'w') as hf:
            hf.create_dataset('nameEn_embeddings',  data = nameEn_embeddings, compression = 'gzip')

        _property_name = 'label'
        train_entities_data[_property_name].fillna('<UNK>', inplace = True)

        property_numpy = train_entities_data[_property_name].to_numpy(copy = True)

        label_embeddings = np.array([get_ft_embedding(x, model) for x in property_numpy], dtype = np.float64)

        # save as h5py
        with h5py.File(os.path.join(data_folder, 'label_embeddings_file.h5'), 'w') as hf:
            hf.create_dataset('label_embeddings',  data = label_embeddings, compression = 'gzip')

        if 'sum' == label_nameEn_merge:
            sum_label_nameEn_embeddings = np.add(nameEn_embeddings, label_embeddings)
            print(f'sum_label_nameEn_embeddings.shape {sum_label_nameEn_embeddings.shape}')
            label_nameEn_emb = sum_label_nameEn_embeddings
            with h5py.File(os.path.join(data_folder, 'sum_label_nameEn_embeddings_file.h5'), 'w') as hf:
                hf.create_dataset('sum_label_nameEn_embeddings',  data = sum_label_nameEn_embeddings, compression = 'gzip')

        elif 'mean' == label_nameEn_merge:
            mean_label_nameEn_embeddings = (np.add(nameEn_embeddings, label_embeddings) / 2.0)
            label_nameEn_emb = mean_label_nameEn_embeddings
            print(f'mean_label_nameEn_embeddings.shape {mean_label_nameEn_embeddings.shape}')
            with h5py.File(os.path.join(data_folder, 'mean_label_nameEn_embeddings_file.h5'), 'w') as hf:
                hf.create_dataset('mean_label_nameEn_embeddings',  data = mean_label_nameEn_embeddings, compression = 'gzip')


    if gen_location_emb:

        # store lon, lat in tensor with shape (batch_size, input_loc_dim = 2)
        coordinates = train_entities_data[['lon', 'lat']].to_numpy()
        
        # convert to tensor
        coordinates_tensor = torch.tensor(coordinates, dtype = torch.float64)
        print(f'coordinates_tensor.shape {coordinates_tensor.shape}')

        # loc_feat: (batch_size, 1, input_loc_dim = 2)
        loc_feat = torch.unsqueeze(coordinates_tensor, dim=1)
        loc_feat = loc_feat.cpu().data.numpy()

        spa_enc = TheoryGridCellSpatialRelationEncoder(spa_embed_dim = 300)
        # loc_embed: torch.Tensor(), (batch_size, 1, spa_embed_dim = num_filts)
        loc_embed = spa_enc(loc_feat)

        # loc_emb: torch.Tensor(), (batch_size, spa_embed_dim = num_filts)
        loc_emb = loc_embed.squeeze(1)

        location_embeddings = loc_emb.numpy()
        print(f'location_embeddings.shape {location_embeddings.shape}')

        # save location embedding as hdf5
        with h5py.File(os.path.join(data_folder, 'location_embeddings_file.h5'), 'w') as hf:
            hf.create_dataset('location_embeddings',  data = location_embeddings, compression = 'gzip')

    ##
    # Input embedding after merging namelabel, location and type embedding
    ##
    if gen_labelNameEn_emb and gen_location_emb and gen_type_emb:

        # Concatenate embeddings
        input_embeddings = np.concatenate((
            location_embeddings,
            label_nameEn_emb,
            type_FT_embeddings),
        axis = 1)
        print(f'input_embeddings.shape {input_embeddings.shape}')

        # save as h5py
        with h5py.File(os.path.join(data_folder, 'input_embeddings_file.h5'), 'w') as hf:
            hf.create_dataset('input_embeddings',  data = input_embeddings, compression = 'gzip')


    ##
    # r_candidates_dict
    ##

    # load entity_idx
    with open(os.path.join(data_folder, 'entity_idx.pickle'), 'rb') as handle:
        entity_idx = pickle.load(handle)

    # prepare dictionary
    r_candidates_dict = dict.fromkeys(relation_uri_idx_dict.values(), [])
    for relation, candidate_type in tqdm(relation_type_mapping.items()):
        uri_list = train_entities_data['entity_uri'][train_entities_data['type'] == candidate_type].tolist()
        candidate_indices = get_indices_from_uri_list(uri_list, entity_idx)
        r_candidates_dict[relation_uri_idx_dict[relation]] = candidate_indices

    # save as pkl file
    with open(os.path.join(data_folder,'r_candidates_dict.pickle'), 'wb') as handle:
        pickle.dump(r_candidates_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ##
    # Fasttext embedding of literal value of spatial property (used in USLP space)
    ##
    if gen_spatialprop_emb:

        # read test triples
        test_inductive = pd.read_csv(
            os.path.join(data_folder, 'test.txt'),
            sep = '\t',
            names = ['subject', 'predicate', 'object']
        )
        print(f'test_inductive.shape {test_inductive.shape}')

        # use literal values collected in groundtruth
        # Note - if using another dataset, need to query KG endoint to collect these literals
        spo_literal_value_df = pd.read_csv(
            os.path.join(data_folder,'spo_plus_literal.zip'), # check and replace
            names = ['subject', 'predicate', 'object', 'literal_object']
            )
        print(f'spo_literal_value_df.shape {spo_literal_value_df.shape}')

        # merge test_inductive with spo_literal_value_df
        test_inductive_with_literals = test_inductive.merge(
            spo_literal_value_df,
            on = ['subject', 'predicate', 'object'],
            how = 'left'
        ).drop_duplicates().reset_index(drop = True)
        print(f'test_inductive_with_literals.shape {test_inductive_with_literals.shape}')

        # prepare FT embedding of literal values
        # triple idx corresponds to literal embedding idx
        _property_name = 'literal_object'
        test_inductive_with_literals[_property_name] = test_inductive_with_literals[_property_name].astype(str)
        property_numpy = test_inductive_with_literals[_property_name].to_numpy(copy = True)
        literal_value_embeddings = np.array([get_ft_embedding(x, model) for x in property_numpy], dtype = np.float64)
        print(f'literal_value_embeddings.shape {literal_value_embeddings.shape}')

        # save as h5py
        with h5py.File(os.path.join(data_folder, 'test_set_literal_value_embeddings_file.h5'), 'w') as hf:
            hf.create_dataset('test_set_literal_value_embeddings',  data = literal_value_embeddings, compression = 'gzip')
        

    ##
    # Dynamic embedding
    ##
    if gen_dynamic_emb:
        model = SentenceTransformer('all-mpnet-base-v2')
        
        # load entity_idx
        with open(os.path.join(data_folder, 'entity_idx.pickle'), 'rb') as handle:
            entity_idx = pickle.load(handle)

        # read literal properties files
        literal_triples = pd.read_csv(
            os.path.join(data_folder, hetero_prop_filename),
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
        print(f'sentence_embeddings.shape {sentence_embeddings.shape}')

        # store embedding in corresponding place in numpy array
        n_entities = len(entity_idx)
        emb_dim = 768 # SBERT emb dim.
        entity_tag_value_SBERT_embeddings = np.zeros((n_entities, emb_dim))
        for idx in tqdm(range(sentence_embeddings.shape[0])):
            entity_idx = literal_triples_modified['uri_idx'][idx]
            entity_tag_value_SBERT_embeddings[entity_idx] = sentence_embeddings[idx]


        # save embedding array
        with h5py.File(os.path.join(data_folder, 'entity_tag_value_SBERT_embeddings_file.h5'), 'w') as hf:
            hf.create_dataset('entity_tag_value_SBERT_embeddings',  data = entity_tag_value_SBERT_embeddings, compression = 'gzip')









