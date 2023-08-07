import numpy as np
import pandas as pd
import math
from collections import defaultdict
import pdb

from torch import empty, matmul, tensor
import torch
from torch.nn import Parameter, Module
import torch.nn.functional as F
from torch.nn.functional import normalize
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



import os
import logging
import json
from tqdm.autonotebook import tqdm
import h5py
import pickle
from abc import ABC, abstractmethod
from prettytable import PrettyTable


from Constants import ModeType, BatchType, train_metrics, Optimizers
from data_handling import DataManager, TrainDataset, TestDataset
from models import SSLP

from utilities import (
    triple_uri_to_idx, set_logger, log_metrics, plot_losses, set_random_seed,
    load_model, save_model, parse_args, override_config
)

def count_parameters(model):
    '''
    Function to print learnable parameters
    '''
    table = PrettyTable(['Modules', 'Parameters'])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    return table, total_params
    


if '__main__' == __name__:

    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    print(device)


    args = parse_args()
    set_random_seed(1)

    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be specified.')

    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be specified.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Write logs to checkpoint and console
    set_logger(args)

    # data - load embeddings, entity/relation dictionaries
    # new tranductive split data manager
    data_dir = '../'
    data_manager = DataManager(
        device = device,
        data_folder = os.path.join(data_dir, args.data_path),
        use_key_value_emb = True,
        use_relation_emb = True,
        use_hierarchy_info = (0.0 < args.hierarchy_weight)
    )
    # dummy_data_manager = DataManager(args.data_path, test_on_slice = 5)

    num_entity = len(data_manager.entity_dict)
    num_relation = len(data_manager.relation_dict)

    logging.info('Data Path: {}'.format(args.data_path))
    logging.info('Num Entity: {}'.format(num_entity))
    logging.info('Num Relation: {}'.format(num_relation))

    logging.info('Num Train: {}'.format(len(data_manager.train_data)))
    logging.info('Num Valid: {}'.format(len(data_manager.valid_data)))
    logging.info('Num Test: {}'.format(len(data_manager.test_data)))

    # initialise model
    if 0.0 < args.hierarchy_weight:
        model = SSLP(
            device = device,
            score_function = args.score_f,
            without_attn = args.without_attn,
            without_static_emb = args.without_static_emb,
            without_dynamic_emb = args.without_dynamic_emb,
            num_entity = num_entity,
            num_relation = num_relation,
            input_embeddings = data_manager.input_embeddings,
            literal_embeddings = data_manager.entity_tag_value_embedding,
            relation_embeddings = data_manager.rel_embeddings,
            entity_hierarchy_level = data_manager.entity_hierarchy_level,
            r_direction = data_manager.r_direction,
            gamma = args.gamma,
            modulus_weight = args.modulus_weight,
            phase_weight = args.phase_weight,
            hierarchy_weight = args.hierarchy_weight
        ).to(device)
    else:
        model = SSLP(
            device = device,
            score_function = args.score_f,
            without_attn = args.without_attn,
            num_entity = num_entity,
            num_relation = num_relation,
            input_embeddings = data_manager.input_embeddings,
            literal_embeddings = data_manager.entity_tag_value_embedding,
            relation_embeddings = data_manager.rel_embeddings,
            # entity_hierarchy_level = data_manager.entity_hierarchy_level,
            # r_direction = data_manager.r_direction,
            gamma = args.gamma,
            modulus_weight = args.modulus_weight,
            phase_weight = args.phase_weight,
            hierarchy_weight = args.hierarchy_weight
        ).to(device)


    logging.info('Model Parameter Configuration:')
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
    table, total_params = count_parameters(model)
    logging.info(f'Total Trainable Params: {total_params}')
    logging.info(table)
    logging.info(f'Type based negative sampling: {args.with_type_sampler}, sample size: {args.negative_sample_size}')
    logging.info(f'Score Function: {args.score_f}')
    if args.without_attn: logging.info(f'Attention Component ablation: {args.without_attn}')
    if args.without_static_emb: logging.info(f'Without static embedding: {args.without_static_emb}')
    if args.without_dynamic_emb: logging.info(f'Without dynamic embedding: {args.without_dynamic_emb}')
    if args.without_layer_norm: logging.info(f'Without layer norm: {args.without_layer_norm}')
    logging.info(f'Hierarchy weight: {args.hierarchy_weight}')

    if args.do_train:

        train_dataloader = DataLoader(
            TrainDataset(
                data_manager,
                args.with_type_sampler,
                args.negative_sample_size,
                BatchType.TAIL_BATCH
            ),
            batch_size = args.batch_size,
            shuffle = True,
            # num_workers=max(1, args.cpu_num // 2),
            collate_fn = TrainDataset.collate_fn
        )

        current_learning_rate = args.learning_rate
        max_steps = args.max_steps
        warm_up_steps = max_steps // 2
        if Optimizers.ADAGRAD.value == args.optimizer:
            optimizer = torch.optim.Adagrad(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr = current_learning_rate
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr = current_learning_rate
            )

        if args.init_checkpoint:
            logging.info('Loading checkpoint %s...' % args.init_checkpoint)
            checkpoint = torch.load(os.path.join(args.init_checkpoint, f"checkpoint_{args.init_checkpoint_step}"))
            init_step = checkpoint['step']
            model.load_state_dict(checkpoint['model_state_dict'])
            if args.do_train:
                current_learning_rate = checkpoint['current_learning_rate']
                warm_up_steps = checkpoint['warm_up_steps']
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            logging.info('Randomly Initializing Model...')
            init_step = 0
        
        step = init_step
        logging.info('Start Training...')
        logging.info('init_step = %d' % init_step)
        if not args.do_test:
            logging.info('learning_rate = %d' % current_learning_rate)
        logging.info('batch_size = %d' % args.batch_size)
        logging.info('hidden_dim = %d' % args.hidden_dim)
        logging.info('gamma = %f' % args.gamma)
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
        logging.info('learning_rate = %f' % (current_learning_rate))

    ###
    # Training
    ###
    if args.do_train:
        training_logs = []
        persisted_training_logs = []
        for step in tqdm(range(init_step, args.max_steps)):
            log = model.train_step(
                model,
                optimizer,
                train_dataloader,
                args
            )

            training_logs.append(log)
            persisted_training_logs.append(log)

            if step >= warm_up_steps:
                if not args.no_decay:
                    current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr = current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3

            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                # save_model(model, optimizer, save_variable_list, args)
                
            if step % args.log_steps == 0:
                metrics = {}
                for metric in train_metrics:
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, metrics, training_logs, args)
                training_logs = []
            
            if args.do_valid and 0 == step % args.valid_steps:
                logging.info('Evaluating on Valid Dataset...')
                metrics, logs = model.test_step(model, data_manager, ModeType.VALIDATION, args)
                log_metrics(ModeType.VALIDATION, step, metrics, logs, args, f'validation_logs')

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }

        # save losses as pkl file
        with open(os.path.join(args.save_path, 'train_losses.pkl'), 'wb') as f:
            pickle.dump(persisted_training_logs, f)
        # plot_losses(persisted_training_logs, args.save_path)
        save_model(model, optimizer, save_variable_list, args)

    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics, logs = model.test_step(model, data_manager, ModeType.VALIDATION, args)
        log_metrics(ModeType.VALIDATION, step, metrics, logs, args, 'val_error_analysis')

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics, logs = model.test_step(model, data_manager, ModeType.TEST, args)
        log_metrics(ModeType.TEST, step, metrics, logs, args, 'test_error_analysis')

