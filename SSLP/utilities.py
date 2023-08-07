import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

import logging
import os
import json
import argparse
import random

from Constants import ModeType, error_analysis_variables, train_metrics

def parse_args(args = None):
    '''
    Argument parser
    '''
    parser = argparse.ArgumentParser(
        description = 'SSLP Model',
        usage = 'main.py [<args>] [-h | --help]'
    )

    parser.add_argument('--do_train', action = 'store_true')
    parser.add_argument('--do_valid', action = 'store_true')
    parser.add_argument('--do_test', action = 'store_true')
    parser.add_argument('--data_path', type = str, default = None)
    
    parser.add_argument('-n', '--negative_sample_size', default = 128, type = int)
    parser.add_argument('-d', '--hidden_dim', default = 343, type = int)
    parser.add_argument('-g', '--gamma', default = 12.0, type = float)
    parser.add_argument('-a', '--adversarial_temperature', default = 1.0, type=float)
    parser.add_argument('-b', '--batch_size', default = 64, type = int)
    parser.add_argument('--test_batch_size', default = 1, type = int, help = 'valid/test batch size')
    parser.add_argument('-mw', '--modulus_weight', default = 1.0, type = float)
    parser.add_argument('-pw', '--phase_weight', default = 0.5, type = float)
    parser.add_argument('-hw', '--hierarchy_weight', default = 0.0, type = float)
    parser.add_argument('-reg', '--regularisation', default = 0.0, type = float)

    parser.add_argument('-optmz', '--optimizer', default = None, type = str)
    parser.add_argument('-lr', '--learning_rate', default = 0.0001, type = float)
    # parser.add_argument('-cpu', '--cpu_num', default = 10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default = None, type = str, help = 'checkpoint path')
    parser.add_argument('-initstep', '--init_checkpoint_step', default = None, type = int, help = 'checkpoint step')
    parser.add_argument('-save', '--save_path', default = None, type = str)
    parser.add_argument('--max_steps', default = 100000, type = int, help = 'no. of epochs')

    parser.add_argument('--save_checkpoint_steps', default = 10000, type = int)
    parser.add_argument('--valid_steps', default = 10000, type = int)
    parser.add_argument('--log_steps', default = 100, type = int, help = 'train log every xx steps')
    parser.add_argument('--test_log_steps', default = 1000, type = int, help = 'valid/test log every xx steps')

    parser.add_argument('--no_decay', action = 'store_true', help = 'Learning rate does not decay')
    parser.add_argument('--with_type_sampler', action = 'store_true', help = 'Apply type based sampling')
    parser.add_argument('--score_f', default = 'HAKE', help = 'Score function')
    parser.add_argument('--without_attn', default = False, help = 'Flag to perform ablation on attention & FC layer')
    parser.add_argument('--without_static_emb', action = 'store_true', help = 'Do not use static embeddings')
    parser.add_argument('--without_dynamic_emb', action = 'store_true', help = 'Do not use dynamic embeddings')
    parser.add_argument('--without_layer_norm', action = 'store_true', help = 'Do not use layer norm')
    
    return parser.parse_args(args)

def set_random_seed(random_seed = None):
    '''
    Using random seed for numpy and torch
    '''
    if(random_seed is None):
        random_seed = 1
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    return


def override_config(args):
    '''
    Override model and data configuration with init_checkpoint config
    '''

    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as f:
        args_dict = json.load(f)

    args.data_path = args_dict['data_path']
    args.hidden_dim = args_dict['hidden_dim']
    args.test_batch_size = args_dict['test_batch_size']



def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path, 'train.log')
    else:
        log_file = os.path.join(args.save_path, 'test.log')
    logging.basicConfig(
        format = '%(asctime)s %(levelname)-8s %(message)s',
        level = logging.INFO,
        datefmt = '%Y-%m-%d %H:%M:%S',
        filename = log_file,
        filemode = 'w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)



def log_metrics(mode, step, metrics, logs, args, error_analysis_file_name = None):
    '''
    Print the evaluation logs
    '''

    # print performance metrics in log file
    for metric in metrics:
        # print(mode, metric, step, metrics[metric])
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

    # save triples/scores in df for error analysis
    if ModeType.TRAIN != mode and error_analysis_file_name:
        logs_df = pd.DataFrame(logs)
        save_file_name = os.path.join(args.save_path, f"{error_analysis_file_name}_{step}.csv")
        logs_df.to_csv(save_file_name, index = False)



def triple_uri_to_idx(triples_path, entity_uri_idx_dict, relation_uri_idx_dict):

    data_idxs = []
    with open(triples_path, 'r') as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            data_idxs.append((entity_uri_idx_dict[head], relation_uri_idx_dict[relation], entity_uri_idx_dict[tail]))

    return data_idxs

def negative_tail_sampler(h_idx, r_idx, t_idx, r_candidates_dict, n_entities, n_samples):
    '''
    Sample negative tails of two types:
    1. tails of same rdf:type as true tail eg. countries when matching wkgs:country
    2. tails of different rdf:type
    :param h_idx, r_idx, t_idx: head, relation, tail idx of triple
    :param r_candidates_dict: dictionary {<relation_id>: [<list of candidates of relation <type>>]}
    :param n_entities
    :param n_samples: no. of negative samples
    :return tensor of entity indices with 1 and 2 type negative tails and true t_idx
    '''

    # negative tails of same rdf:type
    same_type_candidates = r_candidates_dict.get(r_idx)
    sampling_ratio = 0.5 if 0 != len(same_type_candidates) else 1

    # negative tails of different rdf:type
    all_entities_indices = torch.arange(n_entities)
    sampling_weights = torch.ones_like(all_entities_indices).float()
    sampling_weights[t_idx] = 0
    
    if 0 != len(same_type_candidates):
        sampling_weights[same_type_candidates] = 0
    num_samples = n_samples if 1 == sampling_ratio else (n_samples - (n_samples // 2))
    sampled_tails_2 = all_entities_indices[torch.multinomial(sampling_weights, num_samples = num_samples, replacement = True)]
    if 0 == len(same_type_candidates):
        return sampled_tails_2

    same_type_candidates = torch.tensor(same_type_candidates)
    sampling_weights = torch.ones_like(same_type_candidates).float()
    # remove true index from negative candidates
    true_tail_idx = (same_type_candidates == t_idx).nonzero(as_tuple=True)[0]
    sampling_weights[true_tail_idx] = 0
    sampled_tails_1 = same_type_candidates[torch.multinomial(sampling_weights, num_samples = n_samples // 2, replacement = True)]
    
                                                            
    # return torch.cat((sampled_tails_1, sampled_tails_2, torch.tensor([t_idx])), dim = 0)
    return torch.cat((sampled_tails_1, sampled_tails_2), dim = 0)

    
    

def save_model(model, optimizer, save_variable_list, args):
    '''
    Save model, optimizer and additional variables
    '''

    # save all args as dictionary in config.json
    if args:
        args_dict = vars(args)
        with open(os.path.join(args.save_path, 'config.json'), 'w') as f:
            json.dump(args_dict, f, indent = 4)

    # save model and optimizer
    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, f"checkpoint_{save_variable_list['step']}")
    )

    entity_static_embeddings = model.entity_static_embeddings.weight.detach().cpu().numpy()
    entity_dynamic_embeddings = model.entity_dynamic_embeddings.weight.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, f"entity_static_embeddings_{save_variable_list['step']}"),
        entity_static_embeddings
    )
    np.save(
        os.path.join(args.save_path, f"entity_dynamic_embeddings_{save_variable_list['step']}"),
        entity_dynamic_embeddings
    )

    relation_embeddings = model.relation_embeddings.weight.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, f"relation_embeddings_{save_variable_list['step']}"),
        relation_embeddings
    )

# load saved model and embeddings
def load_model(save_path, step, model, optimizer):

    checkpoint = torch.load(os.path.join(save_path, f'checkpoint_{step}'))

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # entity_embedding = np.load(os.path.join(save_path, 'entity_embedding.npy'))
    # relation_embedding = np.load(os.path.join(save_path, 'relation_embedding.npy'))

    return model, optimizer

def plot_losses(training_logs, save_path):
    '''
    Plot losses
    '''
    plt.style.use('seaborn')
    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(18,5)

    metrics_list_dict = {}
    colors = ['blue', 'brown', 'red']
    for i, metric in enumerate(train_metrics):
        metrics_list_dict[metric] = [log[metric] for log in training_logs]

        ax[0].plot(metrics_list_dict[metric], c = colors[i], label = f'{metric}', linewidth = 3, alpha = 0.5)
        ax[0].legend(loc = 'best')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Self-Adversarial Negative Sampling Loss')
        ax[0].set_title(f'Training Loss (linear-scale)')

        ax[1].plot(metrics_list_dict[metric], c = colors[i], label = f'{metric}', linewidth = 3, alpha = 0.5)
        ax[1].legend(loc = 'best')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Self-Adversarial Negative Sampling Loss')
        ax[1].set_yscale('log')
        ax[1].set_title(f'Training Loss (log-scale)')

        # plt.show()
    plt.savefig(os.path.join(save_path, 'loss_plot'))

