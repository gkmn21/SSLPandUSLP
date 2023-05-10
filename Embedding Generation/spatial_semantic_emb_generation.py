"""
Script to generate Spatial and Semantic Embeddings for SSLP
"""

import glob
import os.path
import re
import csv

import pandas as pd
import numpy as np


from urllib.error import HTTPError, URLError
from SPARQLWrapper import SPARQLWrapper, JSON, POST, GET, POSTDIRECTLY, CSV

from io import StringIO
import math
import re
from tqdm import tqdm
import time

import sys
import pickle

import torch
import pygeohash as gh
from haversine import haversine_vector, Unit

import fasttext 
import fasttext.util
import pdb

import h5py


import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import math

from location_encoder import TheoryGridCellSpatialRelationEncoder

#######
# Embedding Generation Utilities
#######

# SPARQL Extraction Utilities
def create_sparql_client(sparql_endpoint, http_query_method = GET, result_format= JSON):
    sparql_client = SPARQLWrapper(sparql_endpoint)

    sparql_client.setMethod(http_query_method)
    sparql_client.setReturnFormat(result_format)

    return sparql_client

# Convert SPARQL results into a Pandas data frame
def sparql2dataframe(json_sparql_results):
    cols = json_sparql_results['head']['vars']
    out = []
    for row in json_sparql_results['results']['bindings']:
        item = []
        for c in cols:
            item.append(row.get(c, {}).get('value'))
        out.append(item)
    return pd.DataFrame(out, columns=cols)

def query_sparql(query, sparql_client):
    sparql_client.setQuery(query)
    result_object = sparql_client.query()
    if sparql_client.returnFormat == JSON:
        return result_object._convertJSON()
    return result_object.convert()


##
# Cleaning coordinates and geohash utilitites code
##
def get_geohash(row):
    '''
    Calculate geohash
    '''
    return gh.encode(row['lat'], row['lon'], precision = 6)

def clean_coordinates(df):
    '''
    :param df - input dataframe
    return clean_df - dataframe with separate columns for 'lat' and 'lon'
    '''
    # applying regex to remove 'Point(' and ')'
    pattern = '(POINT\()|(\))'
    compiled_pattern = re.compile(pattern)
    df['lon_lat'] = df['coordinates'].apply(lambda value: re.sub(compiled_pattern, '', value).split(' '))

    # creating separate columns for lon and lat and convert lat lon columns to float
    df[['lon', 'lat']] = pd.DataFrame(df['lon_lat'].to_list(), index = df.index)
    df['lat'] = df['lat'].astype('float')
    df['lon'] = df['lon'].astype('float')

    df.drop(['lon_lat'], axis = 1, inplace = True)

    return df


'''
Utility to return fastText embedding of word x
using fastText and for <UNK> token return zero vector
'''
def get_ft_embedding(x, model):
    if x == '<UNK>':
        return np.zeros(300)
    else:
        return model.get_word_vector(x.lower())

def get_indices_from_uri_list(uri_list, entity_idx):
    return [entity_idx[uri] for uri in uri_list]


if '__main__' == __name__:

    # train, valid, test files containing subject, predicate, object triples
    train_filename = 'train_split1.txt'
    valid_filename = 'valid_split1.txt'
    test_filename = 'test_split1.txt'

    # global flags
    generate_geohash_and_cluster_centers = True

    train_data = pd.read_csv(
        train_filename,
        sep = '\t',
        names = ['subject', 'predicate', 'object']
        )
    valid_data = pd.read_csv(
        valid_filename,
        sep = '\t',
        names = ['subject', 'predicate', 'object']
        )
    test_data = pd.read_csv(
        test_filename,
        sep = '\t',
        names = ['subject', 'predicate', 'object']
        )

    print(f'train.shape {train_data.shape}\nvalid.shape {valid_data.shape}\ntest.shape {test_data.shape}')

    # unique entities present in subject & object positions in training set
    unique_entities = np.unique(train_data[['subject', 'object']].values)
    print(f'# of unique entities present in subject & object positions in training set {len(unique_entities)}')

    '''
    Fetch train set unique entities data using SPARQL
    '''
    sparqlview_endpoint = "https://www.worldkg.org/sparql"
    sparqlview_wrapper = create_sparql_client(sparql_endpoint=sparqlview_endpoint)

    # extract all triples for each unique entity
    batch_size = 100
    batches = math.ceil(len(unique_entities)/batch_size)

    results_df_list = []
    for i in tqdm(range(batches)):
        selected_entities = unique_entities[i*batch_size: (i+1)*batch_size] 
        selected_entities_string = ' '.join(selected_entities)
        query  =  """
            PREFIX uom: <http://www.opengis.net/def/uom/OGC/1.0/>
            PREFIX wkgs: <http://www.worldkg.org/schema/>
            PREFIX wkg: <http://www.worldkg.org/resource/>
            PREFIX wd: <http://www.wikidata.org/wiki/>
            PREFIX sf: <http://www.opengis.net/ont/sf#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX geo: <http://www.opengis.net/ont/geosparql#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            SELECT DISTINCT ?entity_uri ?type ?label ?nameEn ?coordinates WHERE {
                VALUES ?entity_uri {%s} .
                ?entity_uri wkgs:spatialObject ?geom.
                ?geom geo:asWKT ?coordinates.
                OPTIONAL { ?entity_uri rdf:type ?type .}
                OPTIONAL { ?entity_uri rdfs:label ?label .}
                OPTIONAL { ?entity_uri wkgs:nameEn ?nameEn .}
            }
        """%(selected_entities_string)
        print(query)
        try:
            results = query_sparql(query,sparqlview_wrapper)
        except Exception as e:
            print(sys.exc_info()[0])
            print(e)

        results_df = sparql2dataframe(results)
        print(results_df.shape)
        results_df_list.append(results_df)


    merged_results = pd.concat(results_df_list, ignore_index = True)

    # checking and removing duplicates
    merged_results = merged_results[~merged_results.duplicated(['entity_uri'], keep = 'first')]
    print(f'Merged Results.shape {merged_results}')

    # clean uri of entity_uri
    pattern = '^http://www.worldkg.org/resource/'
    compiled_pattern = re.compile(pattern)
    merged_results['entity_uri'] = merged_results['entity_uri'].apply(lambda uri: re.sub(compiled_pattern, 'wkg:', uri))

    # clean uri of type
    pattern = '^http://www.worldkg.org/schema/'
    compiled_pattern = re.compile(pattern)
    merged_results['type'] = merged_results['type'].astype('str')
    merged_results['type'] = merged_results['type'].apply(lambda uri: re.sub(compiled_pattern, 'wkgs:', uri))
    merged_results.head()

    # replace string 'None'
    merged_results.replace(to_replace=['None'], value=np.nan, inplace=True)

    # parse coordinates into lat lon column
    entities_data_with_geohash = clean_coordinates(merged_results)
    entities_data_with_geohash

    # save
    entities_data_with_geohash.to_csv('final_approach_data.csv', index = False)

    # generate geohashes and store cluster assignment and store as separate csv
    if generate_geohash_and_cluster_centers:
        tqdm.pandas()
        entities_data_with_geohash['geohash'] = entities_data_with_geohash.progress_apply(get_geohash, axis = 1)
        entities_data_with_geohash.head()

        entities_data_with_geohash['1_precision_cluster_center'] = entities_data_with_geohash['geohash'].apply(lambda x: x[0])
        entities_data_with_geohash['3_precision_cluster_center'] = entities_data_with_geohash['geohash'].apply(lambda x: x[:3])
        entities_data_with_geohash['4_precision_cluster_center'] = entities_data_with_geohash['geohash'].apply(lambda x: x[:4])
        entities_data_with_geohash.to_csv('final_approach_data_with_geohash_and_cluster_center.csv', index = False)

        print(f'entities_data_with_geohash.shape {entities_data_with_geohash.shape}')

    """
    Embedding Generation

    1. label + nameEn FT embedding
    2. location embedding
    3. type FT embedding
    4. Relation name FT embedding
    5. SBERT embedding - using script generate_SBERT_emb.py
    """

    # load fastText model
    fasttext.util.download_model('en', if_exists='ignore')  # English
    model = fasttext.load_model('cc.en.300.bin')


    '''FT embedding of entity type'''
    # fill unknown type with <UNK>
    entities_data_with_geohash['type'].fillna('<UNK>', inplace = True)
    entities_data_with_geohash

    # remove 'wkgs:' prefix from type
    pattern = '^wkgs:'
    compiled_pattern = re.compile(pattern)
    entities_data_with_geohash['type'] = entities_data_with_geohash['type'].apply(lambda uri: re.sub(compiled_pattern, '', uri))


    # list of unique types
    unique_types, type_counts = np.unique(entities_data_with_geohash[['type']].values.flatten(), return_counts = True)
    unique_types.shape

    _property_name = 'type'
    property_numpy = entities_data_with_geohash[_property_name].to_numpy(copy = True)

    type_FT_embeddings = np.array([get_ft_embedding(x, model) for x in property_numpy], dtype = np.float64)

    # save as h5py
    with h5py.File('entity_type_FT_embeddings_file.h5', 'w') as hf:
        hf.create_dataset('entity_type_FT_embeddings',  data = type_FT_embeddings, compression = 'gzip')

    '''FT of relation name'''
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

    relation_names = list(relation_idx.keys())

    # remove 'wkgs:' prefix from type
    pattern = '^wkgs:'
    compiled_pattern = re.compile(pattern)

    # merged_results['type'] = merged_results['type'].astype('str')
    relation_names = [re.sub(compiled_pattern, '', uri) for uri in relation_names]
    relation_names

    relation_names_FT_embeddings = np.array([get_ft_embedding(x, model) for x in relation_names], dtype = np.float64)
    relation_names_FT_embeddings.shape

    # save as h5py
    with h5py.File('relation_names_FT_embeddings_file.h5', 'w') as hf:
        hf.create_dataset('relation_names_FT_embeddings',  data = relation_names_FT_embeddings, compression = 'gzip')

    '''label nameEn sum FT embedding'''
    _property_name = 'nameEn'
    entities_data_with_geohash[_property_name].fillna('<UNK>', inplace = True)
    entities_data_with_geohash[pd.isna(entities_data_with_geohash[_property_name])]
    property_numpy = entities_data_with_geohash[_property_name].to_numpy(copy = True)
    nameEn_embeddings = np.array([get_ft_embedding(x, model) for x in property_numpy], dtype = np.float64)

    # save as h5py
    with h5py.File('nameEn_embeddings_file.h5', 'w') as hf:
        hf.create_dataset('nameEn_embeddings',  data = nameEn_embeddings, compression = 'gzip')

    _property_name = 'label'
    entities_data_with_geohash[_property_name].fillna('<UNK>', inplace = True)
    property_numpy = entities_data_with_geohash[_property_name].to_numpy(copy = True)
    label_embeddings = np.array([get_ft_embedding(x, model) for x in property_numpy], dtype = np.float64)

    # save as h5py
    with h5py.File('label_embeddings_file.h5', 'w') as hf:
        hf.create_dataset('label_embeddings',  data = label_embeddings, compression = 'gzip')

    # sum label nameEn embeddings
    sum_label_nameEn_embeddings = np.add(nameEn_embeddings, label_embeddings)
    print(f'{sum_label_nameEn_embeddings.shape}')

    with h5py.File('sum_label_nameEn_embeddings_file.h5', 'w') as hf:
        hf.create_dataset('sum_label_nameEn_embeddings',  data = sum_label_nameEn_embeddings, compression = 'gzip')

    '''location embeddings'''
    # store lon, lat in tensor with shape (batch_size, input_loc_dim = 2)
    coordinates = entities_data_with_geohash[['lon', 'lat']].to_numpy()
    print(f'{coordinates.shape}')

    # convert to tensor
    coordinates_tensor = torch.tensor(coordinates, dtype = torch.float64)
    print(f'{coordinates_tensor.shape}')


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
    with h5py.File('location_embeddings_file.h5', 'w') as hf:
        hf.create_dataset('location_embeddings',  data = location_embeddings, compression = 'gzip')

    """### Input embedding after merging namelabel, location and type embedding"""
    ##
    # Read entity data again but do not use this for embedding generation as
    # previous preprocessing steps are not applied here
    ##
    entities_data_with_geohash = pd.read_csv('final_approach_data_with_geohash_and_cluster_center.csv')

    print(location_embeddings.shape)
    print(type_FT_embeddings.shape)
    print(sum_label_nameEn_embeddings.shape)

    # Concatenate embeddings
    input_embeddings = np.concatenate((
        location_embeddings,
        sum_label_nameEn_embeddings,
        type_FT_embeddings),
    axis = 1)
    input_embeddings.shape

    # save as h5py
    with h5py.File('input_embeddings_with_sum_file.h5', 'w') as hf:
        hf.create_dataset('input_embeddings',  data = input_embeddings, compression = 'gzip')


    '''entity idx pickle'''
    entity_idx = dict()
    for i, uri in enumerate(entities_data_with_geohash['entity_uri']):
        entity_idx[uri] = i

    with open('entity_idx.pickle', 'wb') as handle:
        pickle.dump(entity_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

    '''r_candidates_dict.pickle for negative sampling'''
    relation_type_mapping = {
        'wkgs:country': 'wkgs:Country',
        'wkgs:isInCountry': 'wkgs:Country',
        'wkgs:addrCountry': 'wkgs:Country',
        'wkgs:addrState': 'wkgs:State',
        'wkgs:addrSuburb': 'wkgs:Suburb',
        'wkgs:addrHamlet':  'wkgs:Hamlet',
        'wkgs:addrProvince': 'wkgs:Province',
        'wkgs:addrDistrict': 'wkgs:District',
        'wkgs:isInCounty': 'wkgs:County',
        'wkgs:capitalCity': 'wkgs:City',
        'wkgs:isInContinent': 'wkgs:Continent'
    }

    # prepare dictionary
    # fill unknown type with <UNK>
    entities_data_with_geohash['type'].fillna('<UNK>', inplace = True)
    r_candidates_dict = dict.fromkeys(relation_idx.values(), [])

    for relation, candidate_type in tqdm(relation_type_mapping.items()):
        uri_list = entities_data_with_geohash['entity_uri'][entities_data_with_geohash['type'] == candidate_type].tolist()
        candidate_indices = get_indices_from_uri_list(uri_list, entity_idx)
        r_candidates_dict[relation_idx[relation]] = candidate_indices


    # save as pkl file
    with open('r_candidates_dict.pickle', 'wb') as handle:
        pickle.dump(r_candidates_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)