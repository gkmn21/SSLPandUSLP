'''
Functions for preprocessing data and matching
'''
import re
import pandas as pd
from haversine import haversine_vector, Unit
import numpy as np
import cudf
import cupy
import cuspatial
import geopandas


def split_literal_string(df, literal_prop_name = 'object', column_name = 'object_cleaned'):
  '''
  :param df - input dataframe
  :param literal_prop_name - name of the column containing literals
  return clean_df - dataframe with literal_prop_name values converted to a list of strings having location names and stored in 'object_cleaned' column
  '''
  # converting literal_prop_name column to string type
  df[literal_prop_name] = df[literal_prop_name].astype('string')

  # split literal_prop_name column values at , or ; and store in 'object_cleaned' column without leading and trailing spaces
  df[column_name] = df[literal_prop_name].str.split('\s*,\s*|\s*;\s*')

  return df
  

def clean_coordinates(df):
  '''
  :param df - input dataframe
  return clean_df - geopandas.GeoSeries with coordinates
  '''
  # applying regex to remove 'Point(' and ')'
  # pattern = '(POINT\()|(\))'
  # compiled_pattern = re.compile(pattern)
  # df['lon_lat'] = df['coordinates'].apply(lambda value: re.sub(compiled_pattern, '', value).split(' '))

  # # creating separate columns for lon and lat and convert lat lon columns to float
  # df[['lon', 'lat']] = pd.DataFrame(df['lon_lat'].to_list(), index = df.index)
  # df['lat'] = df['lat'].astype('float')
  # df['lon'] = df['lon'].astype('float')

  coordinates_geoseries = geopandas.GeoSeries.from_wkt(df['coordinates'])

  return cuspatial.from_geopandas(coordinates_geoseries)


def clean_iso_codes(df, isocodes, column_name):
  '''
  :param df - input dataframe
  return clean_df - dataframe with ISO Code for country name replaced by country name
  '''
  # compiled regex for iso-country names
  pattern = '(^[A-Z]{2}$)'
  compiled_pattern = re.compile(pattern)

  df[column_name].apply(lambda literal_list: replace_isocode_in_literal_list(literal_list, compiled_pattern, isocodes))

  return df

def replace_isocode_in_literal_list(literal_list, compiled_pattern, isocodes):

  for index, literal in enumerate(literal_list):
    # match iso-pattern with each literal
    match = re.match(compiled_pattern, literal)
    if match:
      # find country name & substitute code with country name
      iso_name = isocodes[isocodes['iso2'] == match.group(0)]['name']
      if 0 != len(iso_name):
        literal_list[index] = iso_name.values[0]

  return literal_list


def get_country_name_from_iso(literal, compiled_pattern, isocodes):
  
  country_name = literal
  # match iso-pattern with each literal
  match = re.match(compiled_pattern, literal)
  if match:
    # find country name & substitute code with country name
    iso_name = isocodes[isocodes['iso2'] == match.group(0)]['name']
    if 0 != len(iso_name):
      country_name = iso_name.values[0]

  return country_name



def replace_isocode_in_column(df, column_name, isocodes):
  '''
  replace iso_code with country name in a column(<column_name>) of df
  '''
  # compiled regex for iso-country names
  pattern = '(^[A-Z]{2}$)'
  compiled_pattern = re.compile(pattern)

  df[column_name].apply(lambda literal: get_country_name_from_iso(literal, compiled_pattern, isocodes))

  return df

def spatial_distance_filter(subject_entity, candidate_entities_dataframe, top_n):
  '''
  subject_entity: one subject entity
  candidate_entities_dataframe
  # prev_filter
  :return spatial_distance_mask: boolean mask with top_n closest entities(using haversine distance) set to 1
  '''
  # calculate haversine distance
  subject_entity_coord = (subject_entity['lat'], subject_entity['lon'])

  # index_ = candidate_entities_dataframe[prev_filter].index.values_host
  index_ = candidate_entities_dataframe.index.values_host
  # subject_entity_vector = [subject_entity_coord for _ in candidate_entities_dataframe[prev_filter].index]
  subject_entity_vector = [subject_entity_coord for _ in index_]

  # candidate_entities_vector = [(candidate_entities_dataframe['lat'][i], candidate_entities_dataframe['lon'][i]) for i in candidate_entities_dataframe[prev_filter].index]
  candidate_entities_vector = [(candidate_entities_dataframe['lat'][i], candidate_entities_dataframe['lon'][i]) for i in index_]

  distances_array = haversine_vector(subject_entity_vector, candidate_entities_vector, Unit.KILOMETERS)
  
  # extract indices of top_n smallest distances from distances_array and make mask with 1 at these positions
  closest_entities_indices = np.argpartition(distances_array, top_n)[:top_n]
  spatial_distance_mask = candidate_entities_dataframe.index.isin(candidate_entities_dataframe.index[closest_entities_indices])
  # spatial_distance_mask = candidate_entities_dataframe.index.isin(candidate_entities_dataframe[prev_filter].index[closest_entities_indices])

  # print(f"type spatial {type(spatial_distance_mask)}")
  
  return spatial_distance_mask.get()

def cuspatial_spatial_distance_filter(subject_entity_coordinates, candidate_entities_coordinates, top_n):
  '''
  cuspatial variant of spatial_distance_filter
  subject_entity: one subject entity
  candidate_entities_dataframe
  # prev_filter
  :return spatial_distance_mask: boolean mask with top_n closest entities(using haversine distance) set to 1
  '''
  # calculate haversine distance
  # subject_entity_coord = (subject_entity['lat'], subject_entity['lon'])

  # # index_ = candidate_entities_dataframe[prev_filter].index.values_host
  # index_ = candidate_entities_dataframe.index.values_host
  # # subject_entity_vector = [subject_entity_coord for _ in candidate_entities_dataframe[prev_filter].index]
  # subject_entity_vector = [subject_entity_coord for _ in index_]

  # # candidate_entities_vector = [(candidate_entities_dataframe['lat'][i], candidate_entities_dataframe['lon'][i]) for i in candidate_entities_dataframe[prev_filter].index]
  # candidate_entities_vector = [(candidate_entities_dataframe['lat'][i], candidate_entities_dataframe['lon'][i]) for i in index_]

  # distances_array = haversine_vector(subject_entity_vector, candidate_entities_vector, Unit.KILOMETERS)
  subject_coord = cuspatial.GeoSeries([subject_entity_coordinates for _ in candidate_entities_coordinates.index.values_host])
  distances_in_kms = cuspatial.haversine_distance(
    subject_coord.points.x,
    subject_coord.points.y,
    candidate_entities_coordinates.points.x,
    candidate_entities_coordinates.points.y
  )
  distances_in_kms = cudf.Series(distances_in_kms)
  
  # extract indices of top_n smallest distances from distances_array and make mask with 1 at these positions
  # closest_entities_indices = np.argpartition(distances_in_kms, top_n)[:top_n]

  closest_entities_indices = distances_in_kms.argsort()[:top_n]
  spatial_distance_mask = candidate_entities_coordinates.index.isin(candidate_entities_coordinates.index[closest_entities_indices])
  # spatial_distance_mask = candidate_entities_dataframe.index.isin(candidate_entities_dataframe[prev_filter].index[closest_entities_indices])

  # print(f"type spatial {type(spatial_distance_mask)}")
  
  return spatial_distance_mask

def get_matching_entities(subject_entities, literal_property,  candidate_entities_dataframe, check_contains = False):
  '''
  :param subject_entities: a dataframe containing entities having a literal property which needs to be linked
  :param literal_property: name of the property of subject_entity containing literals
  :param candidate_entities_dataframe: dataframe with uri, rdf:label and wkgs:nameEn values of candidate entities
  :matched_entities_result: a list of boolean masks (for all entites in subject entities) which is true at positions of matched entities from candidate_entities_dataframe
  '''
  matched_entities_result = []


  # iterate over all entities in dataframe and perform linking for each entity
  #for index, subject_entity in subject_entities.to_pandas().iterrows():
  for index, subject_entity in subject_entities.iterrows():
    # if 0 == index % 100:
    #   print(f'Matching entity {index}')
    # iterate over each literal in literal_property of entity
    for literal_value in subject_entity[literal_property]:
      
      # match candidates
      # filter_1 exact case-insensitive exact string match
    #   filter = candidate_entities_dataframe['label'].str.fullmatch('^'+ literal_value +'$', 0) | candidate_entities_dataframe['nameEn'].str.fullmatch('^'+ literal_value +'$', 0)
      filter = (candidate_entities_dataframe['label'].str.lower() ==  literal_value.lower()) | (candidate_entities_dataframe['nameEn'].str.lower() ==  literal_value.lower())
      # filter_2 match using contains if there are no results of filter_1
      if 0 == filter.sum() and check_contains:
        filter = candidate_entities_dataframe['label'].str.contains(literal_value, case = False, regex = False) | candidate_entities_dataframe['nameEn'].str.contains(literal_value, case = False, regex=False)
      
      # filter_3 sort by spatial distance if more than 1 result
      if 1 < filter.sum():
        # mask with 1 at top-n closest values after matching with filter_1
        filter = cuspatial_spatial_distance_filter(subject_entity_coordinates, candidate_entities_coordinates, prev_filter = filter, top_n = 1)

      # if there is atleast one matched candidate, stop matching, end loop
      if (0 < filter.sum()):
        break

    # print(f'type {type(filter)}')
    matched_entities_result.append(filter.to_pandas() if isinstance(filter, cudf.core.series.Series) else filter)

  return matched_entities_result

def get_matching_entities_rowwise(row, literal_property,  candidate_entities_dataframe, subject_entities_coordinates, candidate_entities_coordinates, single_literal_property = False, check_contains = False, search_radius = None, mapping_dictionary = None):
  '''
  Row-wise version of the function `get_matching_entities()`
  :param row: a  row of dataframe containing entities having a literal property which needs to be linked
  :param literal_property: name of the property of subject_entity containing literals
  :param candidate_entities_dataframe: dataframe with uri, rdf:label and wkgs:nameEn values of candidate entities
  :matched_entities_result: a list of boolean masks (for all entites in subject entities) which is true at positions of matched entities from candidate_entities_dataframe
  '''
  # if 0 == (row.name  % 100):
  #   print(f'Matching entity {row.name}')
 
  # initialise filter
  filter = [True] * len(candidate_entities_dataframe)

  if single_literal_property:

    literal_value = row[literal_property]

    # match candidates
    
    # filter filter candidates of  that lie within search_radius
    if search_radius:
      filter = cuspatial_filter_candidates_by_search_radius(subject_entities_coordinates.iloc[row.name], candidate_entities_coordinates, search_radius = search_radius)
    candidate_entities_slice = candidate_entities_dataframe[filter]
    
    # filter exact case-insensitive exact string match
    #  filter = candidate_entities_dataframe['label'].str.fullmatch('^'+ literal_value +'$', 0) | candidate_entities_dataframe['nameEn'].str.fullmatch('^'+ literal_value +'$', 0)
    filter = (candidate_entities_slice['label'].str.lower() ==  literal_value.lower()) | (candidate_entities_slice['nameEn'].str.lower() ==  literal_value.lower())
    candidate_entities_slice = candidate_entities_dataframe[candidate_entities_dataframe.index.isin(filter[filter].index)]

    # filter match using contains if there are no results of filter_1
    if 0 == filter.sum() and check_contains:
      filter = candidate_entities_slice['label'].str.contains(literal_value, case = False, regex = False) | candidate_entities_slice['nameEn'].str.contains(literal_value, case = False, regex=False)
      candidate_entities_slice = candidate_entities_dataframe[candidate_entities_dataframe.index.isin(filter[filter].index)]

    # filter sort by spatial distance if more than 1 result
    if 1 < filter.sum():
      # mask with 1 at top-n closest values after matching with filter_1
      # filter = spatial_distance_filter(row, candidate_entities_slice, prev_filter = filter, top_n = 1)
      candidate_entities_coordinates_slice = candidate_entities_coordinates[candidate_entities_coordinates.index.isin(filter[filter].index)]
      filter = cuspatial_spatial_distance_filter(subject_entities_coordinates.iloc[row.name], candidate_entities_coordinates_slice, top_n = 1)
      
    
    # filter contains slice indices, finally change filter to be of the same length as candidate_entities_df
    if isinstance(filter, cupy.ndarray):
      filter = filter.get()
      filter = candidate_entities_dataframe.index.isin(np.where(filter == True))
    else:
      filter = candidate_entities_dataframe.index.isin(filter[filter].index)
    

    # if a mapping dictionary is supplied eg. for continent
    if (0 == filter.sum()) and (mapping_dictionary is not None):
      filter = candidate_entities_dataframe['candidateEntity'] == mapping_dictionary.get(literal_value)
    
      
  else:

    # iterate over each literal in literal_property of entity
    for literal_value in row[literal_property]:

      # reset filter to match new literal value
      filter = [True] * len(candidate_entities_dataframe)
      
      # match candidates

      # filter filter candidates of  that lie within search_radius
      if search_radius:
        filter = cuspatial_filter_candidates_by_search_radius(subject_entities_coordinates.iloc[row.name], candidate_entities_dataframe, search_radius = search_radius)
      candidate_entities_slice = candidate_entities_dataframe[filter]
      
      # filter exact case-insensitive exact string match
      #  filter = candidate_entities_dataframe['label'].str.fullmatch('^'+ literal_value +'$', 0) | candidate_entities_dataframe['nameEn'].str.fullmatch('^'+ literal_value +'$', 0)
      filter = (candidate_entities_slice['label'].str.lower() ==  literal_value.lower()) | (candidate_entities_slice['nameEn'].str.lower() ==  literal_value.lower())
      candidate_entities_slice = candidate_entities_dataframe[candidate_entities_dataframe.index.isin(filter[filter].index)]

      # filter match using contains if there are no results of filter_1
      if 0 == filter.sum() and check_contains:
        filter = candidate_entities_slice['label'].str.contains(literal_value, case = False, regex = False) | candidate_entities_slice['nameEn'].str.contains(literal_value, case = False, regex=False)
        candidate_entities_slice = candidate_entities_dataframe[candidate_entities_dataframe.index.isin(filter[filter].index)]

      # filter sort by spatial distance if more than 1 result
      if 1 < filter.sum():
        # mask with 1 at top-n closest values after matching with filter_1
        candidate_entities_coordinates_slice = candidate_entities_coordinates[candidate_entities_coordinates.index.isin(filter[filter].index)]
        filter = cuspatial_spatial_distance_filter(subject_entities_coordinates.iloc[row.name], candidate_entities_coordinates_slice, top_n = 1)

      # filter contains slice indices, finally change filter to be of the same length as candidate_entities_df
      if isinstance(filter, cupy.ndarray):
        filter = filter.get()
        filter = candidate_entities_dataframe.index.isin(np.where(filter == True))
      else:
        filter = candidate_entities_dataframe.index.isin(filter[filter].index)

      # if a mapping dictionary is supplied eg. for continent
      if (0 == filter.sum()) and (mapping_dictionary is not None):
        filter = candidate_entities_dataframe['candidateEntity'] == mapping_dictionary.get(literal_value)

      # if there is atleast one matched candidate, stop matching, end loop
      if (0 < filter.sum()):
        break
  
  # need to create new column here as list result is getting expanded in df.apply()
  if isinstance(filter, cudf.core.series.Series):
    row['matched_entities'] = filter.to_pandas()
  elif isinstance(filter, cupy.ndarray):
    row['matched_entities'] = filter.get()
  else:
    row['matched_entities'] = filter
  return row

def print_matching_entities(subject_entities, candidate_entities_data, column_name):
 
  for index, row in subject_entities.iterrows():
    
    # print subject entity
    print('')
    print(f"Subject: {row['subject']} \nObject:{row['column_name']}")
    
    # print candidates using mask
    print('\n Matched Entities')
    mask = row['matched_entities']
    print(candidate_entities_data[mask])
    print("------------------------------------")

def prepare_match_dictionary(subject_entities, candidate_entities_data, predicate_uri):
    
  subject_uri_list = []
  object_uri_list = []

  for index in range(len(subject_entities)):
    # only add entities with a match in dictionary
    mask = subject_entities.iloc[index]['matched_entities']
    if 0 < mask.sum():
      subject_uri_list.append(subject_entities.iloc[index]['subject'])
      matched_entity = candidate_entities_data[mask]['candidateEntity'].values[0]
      object_uri_list.append(matched_entity)
  
  match_dictionary =  {
      'subject': subject_uri_list,
      'predicate': [predicate_uri] * len(subject_uri_list),
      'object': object_uri_list
  }

  return match_dictionary


def filter_candidates_by_search_radius(subject_entity, candidate_entities_dataframe, prev_filter, search_radius = None):
  '''
  subject_entity: one subject entity
  candidate_entities_dataframe
  prev_filter
  search_radius: boolean mask set to `True` for entities within this radius of subject
  :return spatial_distance_mask: boolean mask set to `True` for candidate entities within this radius of subject
  '''
  # calculate haversine distance
  subject_entity_coord = (subject_entity['lat'], subject_entity['lon'])

  index_ = candidate_entities_dataframe[prev_filter].index.values_host
  # subject_entity_vector = [subject_entity_coord for _ in candidate_entities_dataframe[prev_filter].index]
  subject_entity_vector = [subject_entity_coord for _ in index_]

  # candidate_entities_vector = [(candidate_entities_dataframe['lat'][i], candidate_entities_dataframe['lon'][i]) for i in candidate_entities_dataframe[prev_filter].index]
  candidate_entities_vector = [(candidate_entities_dataframe['lat'][i], candidate_entities_dataframe['lon'][i]) for i in index_]

  distances_array = haversine_vector(subject_entity_vector, candidate_entities_vector, Unit.KILOMETERS)
  
  # extract indices of top_n smallest distances from distances_array and make mask with 1 at these positions
  spatial_distance_mask = distances_array < search_radius

  return spatial_distance_mask

def cuspatial_filter_candidates_by_search_radius(subject_entity_coordinates, candidate_entities_coordinates, search_radius = None):
  '''
  cuspatial variant of function - filter_candidates_by_search_radius
  subject_entity: one subject entity
  candidate_entities_dataframe
  prev_filter
  search_radius: boolean mask set to `True` for entities within this radius of subject
  :return spatial_distance_mask: boolean mask set to `True` for candidate entities within this radius of subject
  '''
  # temp column to store subject entity coordinates
  # candidate_entities_coordinates['temp_subject_coord'] = subject_entity_coordinates

  # index_ = candidate_entities_dataframe[prev_filter].index.values_host
  # # subject_entity_vector = [subject_entity_coord for _ in candidate_entities_dataframe[prev_filter].index]
  # subject_entity_vector = [subject_entity_coord for _ in index_]

  # # candidate_entities_vector = [(candidate_entities_dataframe['lat'][i], candidate_entities_dataframe['lon'][i]) for i in candidate_entities_dataframe[prev_filter].index]
  # candidate_entities_vector = [(candidate_entities_dataframe['lat'][i], candidate_entities_dataframe['lon'][i]) for i in index_]

  # distances_array = haversine_vector(subject_entity_vector, candidate_entities_vector, Unit.KILOMETERS)
  subject_coord = cuspatial.GeoSeries([subject_entity_coordinates for _ in candidate_entities_coordinates.index.values_host])
  distances_in_kms = cuspatial.haversine_distance(
    subject_coord.points.x,
    subject_coord.points.y,
    candidate_entities_coordinates.points.x,
    candidate_entities_coordinates.points.y
  )
  distances_in_kms = cudf.Series(distances_in_kms)
  
  # extract indices of top_n smallest distances from distances_array and make mask with 1 at these positions
  spatial_distance_mask = distances_in_kms < search_radius

  return spatial_distance_mask