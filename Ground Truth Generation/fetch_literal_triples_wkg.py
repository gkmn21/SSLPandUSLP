import pandas as pd
from urllib.error import HTTPError, URLError
from SPARQLWrapper import SPARQLWrapper, JSON, POST, GET, POSTDIRECTLY, CSV
import requests

from io import StringIO
import math
import re
from tqdm import tqdm
import time

import sys


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

if __name__ == '__main__':

    sparqlview_endpoint = 'https://www.worldkg.org/sparql'
    sparqlview_wrapper = create_sparql_client(sparql_endpoint = sparqlview_endpoint)

    data = pd.read_csv('spo_triples_iter1.csv')

    # extract all triples for each unique entity
    unique_entities_list = pd.unique(data[['subject', 'object']].values.ravel())

    batch_size = 100
    batches = math.ceil(len(unique_entities_list)/batch_size)

    results_df_list = []
    for i in tqdm(range(batches)):
        selected_entities = unique_entities_list[i*100: (i+1)*100]
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
        # print(query)
        try:
            results = query_sparql(query,sparqlview_wrapper)
        except Exception as e:
            print(sys.exc_info()[0])
            print(e)

        results_df = sparql2dataframe(results)
        # print(results_df.shape)
        results_df_list.append(results_df)


    merged_results = pd.concat(results_df_list, ignore_index = True)

    # drop all rows with spatialObject property
    merged_results = merged_results.loc[merged_results['predicate'] != 'http://www.worldkg.org/schema/spatialObject']

    # clean subject and predicte URIs
    pattern_1 = '^http://www.worldkg.org/resource/'
    compiled_pattern = re.compile(pattern_1)
    merged_results['subject'] = merged_results['subject'].apply(lambda uri: re.sub(compiled_pattern, 'wkg:', uri))
    
    pattern_2 = '^http://www.worldkg.org/schema/'
    compiled_pattern = re.compile(pattern_2)
    merged_results['object'] = merged_results['object'].apply(lambda uri: re.sub(compiled_pattern, 'wkgs:', uri))
    merged_results['predicate'] = merged_results['predicate'].apply(lambda uri: re.sub(compiled_pattern, 'wkgs:', uri))

    pattern_3 = '^http://www.w3.org/2000/01/rdf-schema#label'
    compiled_pattern = re.compile(pattern_3)
    merged_results['predicate'] = merged_results['predicate'].apply(lambda uri: re.sub(compiled_pattern, 'rdfs:label', uri))

    pattern_4 = '^http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
    compiled_pattern = re.compile(pattern_4)
    merged_results['predicate'] = merged_results['predicate'].apply(lambda uri: re.sub(compiled_pattern, 'rdf:type', uri))


    print(f'merged_results.shape {merged_results.shape}')
    # save as csv with compression
    compression_opts = dict(method = 'zip', archive_name = 'literal_triples.csv')
    merged_results.to_csv('literal_triples.zip', index = False, compression = compression_opts)

    