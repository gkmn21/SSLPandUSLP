from enum import Enum

class ModeType(Enum):
  TRAIN = 0
  VALIDATION = 1
  TEST = 2

class BatchType(Enum):
  HEAD_BATCH = 0
  TAIL_BATCH = 1
  SINGLE = 2

class Optimizers(Enum):
  ADAGRAD = 'Adagrad'

class ScoreFunction(Enum):
  HAKE = 'HAKE'
  DISTMULT = 'DistMult'

  
'''
variables which are recorded for each sample and used for
error analysis of test set
'''
error_analysis_variables = [
  'triple',
  'true_tail_score',
  'top_ranked_tail_score',
  'top_ranked_tail_index'
]

test_metrics = [
  'MRR', 'MR', 'HITS@1',
  'HITS@3', 'HITS@5', 'HITS@10'
]
train_metrics = [ 'positive_sample_loss', 'negative_sample_loss', 'loss']


relation_uri_idx_dict = {
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

##
# Hierarchy on type
##
type_hierarchy = {
  'wkgs:Continent': 0,
  'wkgs:Sea': 0,
  'wkgs:Ocean': 0,
  'wkgs:Country': 1,
  'wkgs:Island': 1,
  'wkgs:Region': 1,
  'wkgs:State': 2,
  'wkgs:Province': 2,
  'wkgs:District': 3,
  'wkgs:County': 3,
  'wkgs:Village': 4,
  'wkgs:Town': 4,
  'wkgs:City': 4,
  'wkgs:Locality': 5,
  'wkgs:Hamlet': 5, 
  'wkgs:IsolatedDwelling': 5, 
  'wkgs:Neighbourhood': 5,
  'wkgs:Suburb': 5,
  'wkgs:PlaceFarm': 5,
  'wkgs:Quarter': 5,
  'wkgs:CityBlock': 5,
  'wkgs:Square': 5,
  'wkgs:Islet': 5, 
  'wkgs:Borough': 5,
  'wkgs:PlaceAllotments': 5,
  'wkgs:Plot': 5, 
  'wkgs:Municipality': 5,
  'wkgs:Archipelago': 5
}


# relation rule direction
# +1 => L(head) >= L(tail) eg. Bonn isIn Germany
# -1 => L(head) <= L(tail) eg. Germany capitalCity Berlin
relation_rule_direction = {
  'wkgs:isIn': 1,
  'wkgs:addrPlace': 1,
  'wkgs:isInContinent': 1,
  'wkgs:country': 1,
  'wkgs:isInCountry': 1,
  'wkgs:addrCountry': 1, 
  'wkgs:capitalCity': -1,
  'wkgs:addrState': 1,
  'wkgs:addrDistrict': 1,
  'wkgs:addrProvince': 1,
  'wkgs:isInCounty': 1,
  'wkgs:addrSubdistrict': 1,
  'wkgs:addrSuburb': 1,
  'wkgs:addrHamlet': 1
}