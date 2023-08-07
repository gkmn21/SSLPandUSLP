'''
Script to :
1. fetch entities data from SPARQL endpoint for datasets
2. fetch all triples for each entity from SPARQL endpoint
3. store entity info data in `entities_data_filename` file
4. store heterogeneous properties in `hetero_prop_filename` file
5. prepare entity_idx.pickle
'''

import glob
import os.path
import sys
import re
import csv

import pandas as pd
import numpy as np
import fasttext 
import fasttext.util

from urllib.error import HTTPError, URLError
from SPARQLWrapper import SPARQLWrapper, JSON, POST, GET, POSTDIRECTLY, CSV
from io import StringIO


import math
from tqdm import tqdm
import time

import pygeohash as gh
from haversine import haversine_vector, Unit
import h5py
import pickle
import pdb


from dataprep_utils import (
  parse_args_for_fetch_entities, pairwise_similarity_from_distance_matrix,
  create_sparql_client, sparql2dataframe, query_sparql,
  clean_coordinates, get_geohash
)

if '__main__' == __name__:

  args = parse_args_for_fetch_entities()
  print(args)
  data_folder = args.data_folder
  data_filename = args.data_filename
  entities_data_filename = args.entities_data_filename 
  hetero_prop_filename = args.hetero_prop_filename 
  gen_similarity_matrices = args.gen_similarity_matrices
  sparql_endpoint = 'https://www.worldkg.org/sparql'


  train_data = pd.read_csv(
    os.path.join(data_folder, data_filename),
    sep = '\t',
    names = ['subject', 'predicate', 'object']
  )
  print(f'train_data.shape {train_data.shape}')

  # unique entities in train set/inductive test set
  unique_entities = np.unique(train_data[['subject', 'object']].values)
  print(f'Total unique entities in train_data: {len(unique_entities)}')


  ##
  # Fetch train/inductive test set unique entities data from SPARQL endpoint
  ##
  sparqlview_wrapper = create_sparql_client( sparql_endpoint = sparql_endpoint)

  if entities_data_filename:

    # for each unique entity in train set extract type, label, nameEn, coordinates
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
      try:
        results = query_sparql(query,sparqlview_wrapper)
      except Exception as e:
        print(sys.exc_info()[0])
        print(e)
        raise

      results_df = sparql2dataframe(results)
      # print(f'{results_df.shape}')
      results_df_list.append(results_df)


    merged_results = pd.concat(results_df_list, ignore_index = True)

    # checking and removing duplicates
    merged_results = merged_results[~merged_results.duplicated(['entity_uri'], keep = 'first')]
    print(f'merged_results.shape {merged_results.shape}')

    # clean entity uri
    pattern = '^http://www.worldkg.org/resource/'
    compiled_pattern = re.compile(pattern)
    merged_results['entity_uri'] = merged_results['entity_uri'].apply(lambda uri: re.sub(compiled_pattern, 'wkg:', uri))

    # clean type
    pattern = '^http://www.worldkg.org/schema/'
    compiled_pattern = re.compile(pattern)
    merged_results['type'] = merged_results['type'].astype('str')
    merged_results['type'] = merged_results['type'].apply(lambda uri: re.sub(compiled_pattern, 'wkgs:', uri))

    # replace string 'None'
    merged_results.replace(to_replace = ['None'], value = np.nan, inplace = True)


    # parse coordinates into lat lon column
    clean_coordinates(merged_results)
    print(f'entities dataframe shape {merged_results.shape}')

    ##
    # Add geohashes and cluster assignment
    ##
    tqdm.pandas()
    merged_results['geohash'] = merged_results.progress_apply(get_geohash, axis = 1)

    # create a column to indicate cluster assignment for each precision
    merged_results['1_precision_cluster_center'] = merged_results['geohash'].apply(lambda x: x[0])
    merged_results['3_precision_cluster_center'] = merged_results['geohash'].apply(lambda x: x[:3])
    merged_results['4_precision_cluster_center'] = merged_results['geohash'].apply(lambda x: x[:4])
    print(f'merged_results.shape: {merged_results.shape}')
    print(f'merged_results columns: {merged_results.columns}')

    # save entities data as csv
    merged_results.to_csv(os.path.join(data_folder, entities_data_filename), index = False)

    # create pairwise similarity matrices using geohash centroid distance
    if gen_similarity_matrices:
      print('Generating geohash centroid distance matrices')
      precision1_cluster_center_geohashes = np.unique(merged_results['1_precision_cluster_center'])
      precision1_cluster_lat_lon_list = [(gh.decode(hash)[0],  gh.decode(hash)[1]) for hash in precision1_cluster_center_geohashes]
      precision1_cluster_distance_matrix = haversine_vector(
        precision1_cluster_lat_lon_list,
        precision1_cluster_lat_lon_list,
        Unit.KILOMETERS,
        comb=True
      )
      print(f'precision1_cluster_distance_matrix.shape {precision1_cluster_distance_matrix.shape}')


      precision3_cluster_center_geohashes = np.unique(merged_results['3_precision_cluster_center'])
      precision3_cluster_lat_lon_list = [(gh.decode(hash)[0],  gh.decode(hash)[1]) for hash in precision3_cluster_center_geohashes]
      precision3_cluster_distance_matrix = haversine_vector(
        precision3_cluster_lat_lon_list,
        precision3_cluster_lat_lon_list,
        Unit.KILOMETERS,
        comb=True
      )
      print(f'precision3_cluster_distance_matrix.shape {precision1_cluster_distance_matrix.shape}')
      precision1_similarity_matrix = pairwise_similarity_from_distance_matrix(precision1_cluster_distance_matrix)
      precision3_similarity_matrix = pairwise_similarity_from_distance_matrix(precision3_cluster_distance_matrix)
      
      # save similarity matrix
      with h5py.File(os.path.join(data_folder, 'precision1_similarity_matrix_file.h5'), 'w') as hf:
        hf.create_dataset('precision1_similarity_matrix',  data = precision1_similarity_matrix, compression = 'gzip')

      with h5py.File(os.path.join(data_folder,'precision3_similarity_matrix_file.h5'), 'w') as hf:
        hf.create_dataset('precision3_similarity_matrix',  data = precision3_similarity_matrix, compression = 'gzip')

      precision4_cluster_center_geohashes = np.unique(merged_results['4_precision_cluster_center'])
      precision4_cluster_lat_lon_list = [(gh.decode(hash)[0],  gh.decode(hash)[1]) for hash in precision4_cluster_center_geohashes]
      precision4_cluster_distance_matrix = haversine_vector(
        precision4_cluster_lat_lon_list,
        precision4_cluster_lat_lon_list,
        Unit.KILOMETERS,
        comb=True
      )
      precision4_cluster_distance_matrix = precision4_cluster_distance_matrix.astype(
        dtype = np.float16,
        copy = False
      )
      print(f'precision4_cluster_distance_matrix.shape {precision4_cluster_distance_matrix.shape}')
      precision4_similarity_matrix = pairwise_similarity_from_distance_matrix(precision4_cluster_distance_matrix)
      with h5py.File(os.path.join(data_folder,'precision4_similarity_matrix_file.h5'), 'w') as hf:
        hf.create_dataset('precision4_similarity_matrix',  data = precision4_similarity_matrix, compression = 'gzip')


    ##
    # Prepare entity index and save as pickle
    ##
    entity_idx = dict()
    for i, uri in enumerate(merged_results['entity_uri']):
      entity_idx[uri] = i

    with open(os.path.join(data_folder,'entity_idx.pickle'), 'wb') as handle:
      pickle.dump(entity_idx, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    if hetero_prop_filename:
      
      # extract all triples for each unique entity
      batch_size = 100
      batches = math.ceil(len(unique_entities)/batch_size)

      results_df_list = []
      for i in tqdm(range(batches)):
        selected_entities = unique_entities[i*100: (i+1)*100]
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
          SELECT DISTINCT ?subject ?predicate ?object WHERE {
              VALUES ?subject {%s}.
              ?subject ?predicate ?object.
          }
        """%(selected_entities_string)
        try:
          results = query_sparql(query,sparqlview_wrapper)
        except Exception as e:
          print(sys.exc_info()[0])
          print(e)
          # not raising this exception
          # if the heterogeneous properties are not extracted, continue with collected data

        results_df = sparql2dataframe(results)
        results_df_list.append(results_df)


      merged_results = pd.concat(results_df_list, ignore_index = True)
      # remove duplicate rows from merged_results
      merged_results = merged_results[~merged_results.duplicated(keep = 'first')].reset_index(drop = True)
      print(f'merged_results.shape {merged_results.shape}')
      
      # remove/change uri prefix from triples
      pattern = '^http://www.worldkg.org/schema/'
      compiled_pattern = re.compile(pattern)
      merged_results['predicate'] = merged_results['predicate'].apply(lambda uri: re.sub(compiled_pattern, '', uri))
      pattern = '^http://www.worldkg.org/resource/'
      compiled_pattern = re.compile(pattern)
      merged_results['object'] = merged_results['object'].apply(lambda uri: re.sub(compiled_pattern, 'wkg:', uri))
      pattern = '^http://www.worldkg.org/resource/'
      compiled_pattern = re.compile(pattern)
      merged_results['subject'] = merged_results['subject'].apply(lambda uri: re.sub(compiled_pattern, 'wkg:', uri))

      '''
      Remove following properties from triples as they will be embedded separately and osmLink is not needed
      ['rdfs:label', 'rdf:type', 'wkgs:nameEn', 'wkgs:osmLink', 'wkgs:spatialObject']
      '''
      filter_out_predicates = ['http://www.w3.org/2000/01/rdf-schema#label', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'nameEn', 'osmLink', 'spatialObject']
      merged_results = merged_results[~merged_results['predicate'].isin(filter_out_predicates)]
    
      # remove/change wikidata prefix from object
      pattern = '^http://www.wikidata.org/wiki/'
      compiled_pattern = re.compile(pattern)
      merged_results['object'] = merged_results['object'].apply(lambda uri: re.sub(compiled_pattern, '', uri))
      merged_results.reset_index(drop = True, inplace = True)
      print(f'Key Value df shape {merged_results.shape}')
      
      # save dataframe with heterogeneous properties
      merged_results.to_csv(os.path.join(data_folder, hetero_prop_filename), sep = '\t', index = False, header = False)
      


