'''
Training Utilities
'''
import pickle
from tqdm import tqdm

# from pykeen.models.predict import get_tail_prediction_df, get_all_prediction_df

def save_to_pkl(data, file_name):
  with open(file_name, "wb") as f: 
    pickle.dump(data, f)

def read_from_pkl(file_name):
  with open(file_name, "rb") as f:
    data = pickle.load(f)
  return data

def get_incorrect_tail_predictions(testing_factory, pipeline_result):
  '''
  :param testing_factory
  :param pipeline_result
  compute tail using triples in testing factory and model present in result 
  :return incorrect_predictions: list of incorrectly predicted triples ('h, r, predicted_t, true_t')
  '''

  total_triples = testing_factory.num_triples
#   correct_triples = 0
#   correct_triples_check = 0
  incorrect_triple_predictions = []

  for triple in tqdm(testing_factory.mapped_triples):

    head_label = pipeline_result.training.entity_id_to_label.get(triple[0].item())
    relation_label = pipeline_result.training.relation_id_to_label.get(triple[1].item())

    # predict the tail from the head and relation
    tail_prediction_df = get_tail_prediction_df(
        model = pipeline_result.model,
        head_label = head_label,
        relation_label = relation_label,
        triples_factory = pipeline_result.training
    )

    prediction = tail_prediction_df.nlargest(1, 'score')
    predicted_tail_id = prediction['tail_id'].values[0]
    predicted_tail_label = prediction['tail_label'].values[0]
    

    true_tail_id = triple[2].item()
    true_tail_label = pipeline_result.training.entity_id_to_label.get(true_tail_id)

    
    # if predicted_tail_label == true_tail_label:
    #   correct_triples += 1
    # else:
    if predicted_tail_label != true_tail_label:
      incorrect_triple = (head_label, relation_label, predicted_tail_label, true_tail_label)
      incorrect_triple_predictions.append(incorrect_triple)
      
      
    # if predicted_tail_id == true_tail_id:
    #   correct_triples_check += 1
  
#   print('h, r, predicted_t, true_t')
#   print(incorrect_triple_predictions)
#   print(f'Correct triples {correct_triples}')
#   print(f'Correct triples check {correct_triples_check}')
  print(f'Total triples {total_triples}')
  print(f'Accuracy {correct_triples/total_triples}')

  return incorrect_triple_predictions
