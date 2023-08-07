'''
Utilities for data preparation in Transductive and Inductive setting
'''

import pandas as pd
import numpy as np
import re
import pygeohash as gh
from urllib.error import HTTPError, URLError
from SPARQLWrapper import SPARQLWrapper, JSON, POST, GET, POSTDIRECTLY, CSV
from io import StringIO
import argparse

##
# SPARQL Extraction Utilities
##

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
# Geohash and coordinates handling utilities code
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


def pairwise_similarity_from_distance_matrix(distance_matrix):
    '''
    :param distance_matrix: distance_matrix of cluster centers
    :return pairwise similarity matrix based on distance matrix with values in [0, 1]
    '''

    # convert matrix of distances to values between 0-1 and change distance to similarity
    # i.r d/d_max and the 1 - d/d_max
    # pairwise_similarity_matrix = distance_matrix / np.broadcast_to(distance_matrix.max(axis = 0), distance_matrix.shape)
    # pairwise_similarity_matrix = 1 - pairwise_similarity_matrix

    # return pairwise_similarity_matrix
    return 1 - (distance_matrix / np.broadcast_to(distance_matrix.max(axis = 0), distance_matrix.shape))

##
# Static embedding utilities
##
'''
Utility to return fastText embedding of word x
using fastText and for <UNK> token return zero vector
'''
def get_ft_embedding(x, model):
    if x == '<UNK>':
        return np.zeros(300)
    else:
        return model.get_word_vector(x.lower())

##
# For type-based negative sampling
##
def get_indices_from_uri_list(uri_list, entity_idx):
    return [entity_idx[uri] for uri in uri_list]

def parse_args(args = None):
    '''
    Argument parser
    '''
    parser = argparse.ArgumentParser(
        description = 'Parser for embedding generation script',
        usage = 'generate_SSLP_embeddings.py [<args>] [-h | --help]'
    )
    parser.add_argument('--data_folder', type = str, default = 'Data')
    parser.add_argument('--entities_data_filename', type = str, default = 'final_approach_data.csv')
    parser.add_argument('--hetero_prop_filename', type = str, default = 'KeyValues_of_final_approach_entities.txt')
    parser.add_argument('--gen_labelNameEn_emb', action = 'store_true')
    parser.add_argument('--gen_location_emb', action = 'store_true')
    parser.add_argument('--gen_type_emb', action = 'store_true')
    parser.add_argument('--gen_relationName_emb', action = 'store_true')
    parser.add_argument('--gen_type_and_relation_emb', action = 'store_true', help = 'single embedding file and index of relation and type for USLP')
    parser.add_argument('--gen_hierarchy_data', action = 'store_true', help = 'generation relation direction dict and hierarchy levels')
    parser.add_argument('--gen_dynamic_emb', action = 'store_true')
    parser.add_argument('--gen_spatialprop_emb', action = 'store_true', help = 'generate fastText emb. of literal value of spatial property (for USLP)')
    parser.add_argument('--label_nameEn_merge', type = str, default = None, help = 'sum or mean')
    
    return parser.parse_args(args)

def parse_args_for_fetch_entities(args = None):
    '''
    Argument parser
    '''
    parser = argparse.ArgumentParser(
        description = 'Parser for fetch entities script',
        usage = 'fetch_entities_info.py [<args>] [-h | --help]'
    )
    parser.add_argument('--data_folder', type = str, default = 'Data', help = 'path to folder with train, valid, test data')
    parser.add_argument('--data_filename', type = str, default = 'train.txt', help = 'name of txt file with training triples')
    parser.add_argument('--entities_data_filename', type = str, default = 'final_approach_data.csv')
    parser.add_argument('--hetero_prop_filename', type = str, default = 'KeyValues_of_final_approach_entities.txt')
    parser.add_argument('--gen_similarity_matrices', action = 'store_true', help = 'whether to generate geohash similarity matrices')
    
    return parser.parse_args(args)


