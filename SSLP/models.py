import numpy as np
from abc import ABC, abstractmethod
from tqdm.autonotebook import tqdm
import logging
import pdb

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader



from Constants import BatchType, ModeType, test_metrics, ScoreFunction
from data_handling import TestDataset

# Abstract base class for multiple input embeddings
class KGEModelWithMultiInput(nn.Module, ABC):

    @abstractmethod
    def predict(self, h_static, h_dynamic, r, t_static, t_dynamic, r_direction, head_hierarchy_level, tail_hierarchy_level, batch_type):
        '''
        Different input shapes depending on batch type
        BatchType.SINGLE:
            head_static: [batch_size, 1, hidden_dim]
            head_dynamic: [batch_size, 1, hidden_dim]
            relation: [batch_size, 1, hidden_dim]
            tail_static: [batch_size, 1, hidden_dim]
            tail_dynamic: [batch_size, 1, hidden_dim]
            r_direction: [batch_size, 1, 1]
            head_hierarchy_level: [batch_size, 1, 1]
            tail_hierarchy_level: [batch_size, 1, 1]

        BatchType.TAIL_BATCH:
            head_static: [batch_size, 1, hidden_dim]
            head_dynamic: [batch_size, 1, hidden_dim]
            relation: [batch_size, 1, hidden_dim]
            tail_static: [batch_size, negative_sample_size, hidden_dim]
            tail_dynamic: [batch_size, negative_sample_size, hidden_dim]
            r_direction: [batch_size, 1, 1]
            head_hierarchy_level: [batch_size, 1, 1]
            tail_hierarchy_level: [batch_size, negative_sample_size, 1]
        '''
        pass

    def forward(self, sample, batch_type = BatchType.SINGLE):
        '''
        Given the indexes in `sample`, extract the corresponding embeddings,
        and call predict().
        Params
        batch_type: {SINGLE, TAIL_BATCH}
        sample: different format for different batch types
            - SINGLE: tensor with shape [batch_size, 3]
            - TAIL_BATCH: (positive_sample, negative_sample)
                - positive_sample: tensor with shape [batch_size, 3]
                - negative_sample: tensor with shape [batch_size, negative_sample_size]
        '''
        if batch_type == BatchType.SINGLE:
            h_static = self.entity_static_embeddings(sample[:, 0]).unsqueeze(1)
            h_dynamic = self.entity_dynamic_embeddings(sample[:, 0]).unsqueeze(1)

            relation = self.relation_embeddings(sample[:, 1]).unsqueeze(1)

            t_static = self.entity_static_embeddings(sample[:, 2]).unsqueeze(1)
            t_dynamic = self.entity_dynamic_embeddings(sample[:, 2]).unsqueeze(1)

            # pdb.set_trace()
            if 0.0 < self.hierarchy_weight: 
                r_direction = self.r_direction[sample[:, 1]].unsqueeze(1)
                head_hierarchy_level = self.entity_hierarchy_level[sample[:, 0]].unsqueeze(1)
                tail_hierarchy_level = self.entity_hierarchy_level[sample[:, 2]].unsqueeze(1)
            else:
                r_direction = None
                head_hierarchy_level = None
                tail_hierarchy_level = None


        elif batch_type == BatchType.TAIL_BATCH:

            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            h_static = self.entity_static_embeddings(head_part[:, 0]).unsqueeze(1)
            h_dynamic = self.entity_dynamic_embeddings(head_part[:, 0]).unsqueeze(1)

            relation = self.relation_embeddings(head_part[:, 1]).unsqueeze(1)

            t_static = self.entity_static_embeddings(tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
            t_dynamic = self.entity_dynamic_embeddings(tail_part.view(-1)).view(batch_size, negative_sample_size, -1)

            
            # pdb.set_trace()
            if 0.0 < self.hierarchy_weight:
                r_direction = self.r_direction[head_part[:, 1]].unsqueeze(1)
                head_hierarchy_level = self.entity_hierarchy_level[head_part[:, 0]].unsqueeze(1)
                tail_hierarchy_level = self.entity_hierarchy_level[tail_part.view(-1)].view(batch_size, negative_sample_size, -1)
            else:
                r_direction = None
                head_hierarchy_level = None
                tail_hierarchy_level = None
            
                
        # return score
        return self.predict(h_static, h_dynamic, relation, t_static, t_dynamic, r_direction, head_hierarchy_level, tail_hierarchy_level, batch_type)

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_type = next(iter(train_iterator))

        positive_sample = positive_sample.to(model.device)
        negative_sample = negative_sample.to(model.device)
        subsampling_weight = subsampling_weight.to(model.device)

        # negative scores
        negative_score = model((positive_sample, negative_sample), batch_type=batch_type)

        negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                          * F.logsigmoid(-negative_score)).sum(dim=1)

        # positive scores
        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        positive_sample_loss = - positive_score.mean()
        negative_sample_loss = - negative_score.mean()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        # apply regularisation
        if args.regularisation != 0.0:
            regularisation = args.regularisation * (
                model.entity_static_embeddings.weight.norm(p = 2.0)**2 + model.entity_dynamic_embeddings.weight.norm(p = 2.0)**2 + model.relation_embeddings.weight.norm(p = 2.0)**2
            )
            # pdb.set_trace()
            loss = loss + regularisation

        loss.backward()

        optimizer.step()

        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log


    @staticmethod
    def test_step(model, data_manager, mode, args):
        '''
        Evaluate the model on test or valid datasets
        args = {
            test_batch_size,
            test_log_steps
        }
        '''

        model.eval()

        test_dataloader = DataLoader(
            TestDataset(
                data_manager,
                mode,
                BatchType.TAIL_BATCH
            ),
            batch_size = args.test_batch_size,
            collate_fn = TestDataset.collate_fn
        )

        logs = []

        step = 0
        total_steps = len(test_dataloader)

        with torch.no_grad():
            for positive_sample, negative_sample, filter_bias, batch_type in tqdm(test_dataloader):
                positive_sample = positive_sample.to(model.device)
                negative_sample = negative_sample.to(model.device)
                filter_bias = filter_bias.to(model.device)

                batch_size = positive_sample.size(0)

                score = model((positive_sample, negative_sample), batch_type)
                score += filter_bias

                # sorted indices based on score
                argsort = torch.argsort(score, dim = 1, descending = True)
                if BatchType.TAIL_BATCH == batch_type:
                    positive_arg = positive_sample[:, 2]
                
                for i in range(batch_size):
                
                    rank = (argsort[i, :] == positive_arg[i]).nonzero()
                    assert rank.size(0) == 1

                    ranking = 1 + rank.item()

                    # top_5_tail_indices = argsort[i, :5]
                    # pdb.set_trace()
                    # top_5_tails = [negative_sample[0, idx] for idx in top_5_tail_indices] 
                    logs.append({
                        'MRR': 1.0 / ranking,
                        'MR': float(ranking),
                        'HITS@1': 1.0 if ranking <= 1 else 0.0,
                        'HITS@3': 1.0 if ranking <= 3 else 0.0,
                        'HITS@5': 1.0 if ranking <= 5 else 0.0,
                        'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        'triple': positive_sample[i].cpu().numpy().copy(),
                        'true_tail_score': score[i, positive_arg[i]].item(),
                        'top_ranked_tail_score': score[i, argsort[i, 0]].item(),
                        'top_ranked_tail_index': argsort[i, 0].item()
                        # 'top_5_scores': score[i, :5].cpu().numpy().copy(),
                        # 'top_5_tail_idx': top_5_tails
                    })
                

                # if step % args.test_log_steps == 0:
                #     logging.info('Evaluating the model... ({}/{})'.format(step, total_steps))

                step += 1
            
        metrics = {}
        for metric in test_metrics:
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        # metrics['true_tail_score'] = [log['true_tail_score'] for log in logs]
        # metrics['triple'] = [log['triple'] for log in logs]
        # metrics['top_ranked_tail_score'] = [log['top_ranked_tail_score'] for log in logs]
        # metrics['top_ranked_tail_index'] = [log['top_ranked_tail_index'] for log in logs]

        return metrics, logs



'''
SSLP Model
Apply Attention on tail_combined[t_static + t_dynamic] with q = h_combined concat relation embedding
'''
class SSLP(KGEModelWithMultiInput):
    '''
    input_embeddings: (n_entities, emb_dim)
    literal_embeddings: (n_entities, emb_dim)
    relation_embeddings: (n_relations, emb_dim_rel)
    '''

    def __init__(self,
    device = 'cpu',
    score_function = None,
    without_attn = False,
    without_static_emb = False,
    without_dynamic_emb = False,
    num_entity = None,
    num_relation = None,
    input_embeddings = None,
    literal_embeddings = None,
    relation_embeddings = None,
    entity_hierarchy_level = None,
    r_direction = None,
    gamma = 0.0, modulus_weight = 0.0, phase_weight = 0.0, hierarchy_weight = 0.0):

        super(SSLP, self).__init__()
        self.device = device
        self.score_function = score_function
        self.without_attn = without_attn
        self.without_static_emb = without_static_emb
        self.without_dynamic_emb = without_dynamic_emb
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.epsilon = 2.0
        r_dim = 64 * 3 # rel emb will be proj to this dim
        

        self.pi = 3.14159262358979323846

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad = False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / r_dim]),
            requires_grad = False
        )

        self.entity_static_embeddings = torch.nn.Embedding.from_pretrained(
            input_embeddings,
            freeze = False
        )

        # entity literal embeddings (SBERT sentence embeddings)
        self.entity_dynamic_embeddings = torch.nn.Embedding.from_pretrained(
            literal_embeddings,
            freeze = False
        )

        # FT relation embeddings
        self.relation_embeddings = torch.nn.Embedding.from_pretrained(
            relation_embeddings,
            freeze = False
        )

        # Hierarchy related info
        if 0.0 < hierarchy_weight:
            self.entity_hierarchy_level = entity_hierarchy_level
            self.r_direction = r_direction

        # linear layer for h_static embedding reduce dim from 6** -> 512 -> 256 -> 128
        self.h_static_layer1 = nn.Linear(input_embeddings.shape[1], 512)
        self.h_s_layer_norm1 = nn.LayerNorm(512)
        self.h_static_layer2 = nn.Linear(512, 256)
        self.h_s_layer_norm2 = nn.LayerNorm(256)
        self.h_static_layer3 = nn.Linear(256, 128)
        self.h_s_layer_norm3 = nn.LayerNorm(128)

        # linear layer for h_dynamic embedding 768 -> 512 -> 256
        self.h_dynamic_layer1 = nn.Linear(literal_embeddings.shape[-1], 512)
        self.h_d_layer_norm1 = nn.LayerNorm(512)
        self.h_dynamic_layer2 = nn.Linear(512, 256)
        self.h_d_layer_norm2 = nn.LayerNorm(256)
        self.h_dynamic_layer3 = nn.Linear(256, 128)
        self.h_d_layer_norm3 = nn.LayerNorm(128)

        self.h_combined = nn.Linear(256, 128)
        self.h_c_layer_norm = nn.LayerNorm(128)

        self.h_r_combined = nn.Linear(256, 128)
        self.h_r_layer_norm = nn.LayerNorm(128)


        # MHA for combining t_combined with h_combined + relation
        self.tail_attn_layer = nn.MultiheadAttention(
            embed_dim = 128,
            num_heads = 8,
            batch_first = True,
            device = self.device
        )

        # linear layer for relation
        self.rel_layer1 = nn.Linear(relation_embeddings.shape[-1], 128)
        self.rel_layer_norm = nn.LayerNorm(128)

        # project rel to head/tail_dim *3 since it will be split into 3 chunks for HAKE
        self.rel_proj_layer = nn.Linear(128, 64 * 3)
        self.rel_proj_layer_norm = nn.LayerNorm(64*3)

        self.phase_weight = nn.Parameter(torch.Tensor([[phase_weight * self.embedding_range.item()]]))
        self.modulus_weight = nn.Parameter(torch.Tensor([[modulus_weight]]))
        if 0.0 < hierarchy_weight:
            self.hierarchy_weight = nn.Parameter(torch.Tensor([[hierarchy_weight]]))
        else: self.hierarchy_weight = 0.0
    

    def predict(self, head_static, h_dynamic, rel, tail_static, t_dynamic, r_direction, head_hierarchy_level, tail_hierarchy_level, batch_type):
        '''
        Different input shapes depending on batch type
        BatchType.SINGLE:
            head_static: [batch_size, 1, hidden_dim]
            head_dynamic: [batch_size, 1, hidden_dim]
            relation: [batch_size, 1, hidden_dim]
            tail_static: [batch_size, 1, hidden_dim]
            tail_dynamic: [batch_size, 1, hidden_dim]
            r_direction: [batch_size]
            head_hierarchy_level: [batch_size]
            tail_hierarchy_level: [batch_size]

        BatchType.TAIL_BATCH:
            head_static: [batch_size, 1, hidden_dim]
            head_dynamic: [batch_size, 1, hidden_dim]
            relation: [batch_size, 1, hidden_dim]
            tail_static: [batch_size, negative_sample_size, hidden_dim]
            tail_dynamic: [batch_size, negative_sample_size, hidden_dim]
            r_direction: [batch_size]
            head_hierarchy_level: [batch_size]
            tail_hierarchy_level: [batch_size, negative_sample_size]
        '''

        # h_static refinement
        if not self.without_static_emb:
            h_static = self.h_s_layer_norm3(F.relu(self.h_static_layer3(
                self.h_s_layer_norm2(F.relu(self.h_static_layer2(
                    self.h_s_layer_norm1(F.relu(self.h_static_layer1(head_static)))
                )))
            )))

        # relation embedding refinement
        rel = self.rel_layer_norm(F.relu(self.rel_layer1(rel)))

        # query

        # h_dynamic refinement # (batch, 1, emb_dim)
        if not self.without_dynamic_emb:
            h_dynamic = self.h_d_layer_norm3(F.relu(self.h_dynamic_layer3(
                self.h_d_layer_norm2(F.relu(self.h_dynamic_layer2(
                    self.h_d_layer_norm1(F.relu(self.h_dynamic_layer1(h_dynamic)))
                )))
            )))
        
        if self.without_static_emb:
            head_emb = h_dynamic
        elif self.without_dynamic_emb:
            head_emb = h_static
        else:
            head_emb = torch.cat((h_static, h_dynamic), dim = 2)
            head_emb = self.h_c_layer_norm(F.relu(self.h_combined(head_emb)))
            # print(f'head_emb.shape {head_emb.shape}')

        # t_static refinement
        # (batch, nsample_size, emb_dim)
        if not self.without_static_emb:
            t_static = self.h_s_layer_norm3(F.relu(self.h_static_layer3(
                self.h_s_layer_norm2(F.relu(self.h_static_layer2(
                    self.h_s_layer_norm1(F.relu(self.h_static_layer1(tail_static)))
                )))
            )))

        # t_dynamic refinement
        # (batch, nsample_size, emb_dim)
        if not self.without_dynamic_emb:
            t_dynamic = self.h_d_layer_norm3(F.relu(self.h_dynamic_layer3(
                self.h_d_layer_norm2(F.relu(self.h_dynamic_layer2(
                    self.h_d_layer_norm1(F.relu(self.h_dynamic_layer1(t_dynamic)))
                )))
            )))
        
        if self.without_static_emb:
            tail_emb = t_dynamic
        elif self.without_dynamic_emb:
            tail_emb = t_static
        else:
            tail_emb = torch.cat((t_static, t_dynamic), dim = 2)
            tail_emb = self.h_c_layer_norm(F.relu(self.h_combined(tail_emb)))

        if not self.without_attn:
            # loop over batches and apply attention to tail embedding by taking
            # negative_sample_size as batch dimension for attention
            # q - (nsample_size, 1, emb_dim); k,v - (nsample, nkeys, emb_dim)
            h_r_query = self.h_r_layer_norm(F.relu(self.h_r_combined(torch.cat((head_emb, rel), dim = 2))))
            h_r_query = h_r_query.unsqueeze(2)
            attn_output_list = []
            for batch_idx in range(tail_emb.size(0)):
                attn_result, _ = self.tail_attn_layer(
                    query = h_r_query[batch_idx].expand((tail_emb.size(1), h_r_query.size(2), h_r_query.size(3))),
                    key = tail_emb[batch_idx].unsqueeze(1),
                    value = tail_emb[batch_idx].unsqueeze(1)
                )

                attn_result = attn_result.squeeze(1)
                # creating new batch dim to concatenate along
                attn_result = attn_result.unsqueeze(0)
                attn_output_list.append(attn_result)
            tail_emb = torch.cat(attn_output_list)



        # Scoring
        if ScoreFunction.HAKE.value == self.score_function:
                
            ##
            # HAKE score
            ##
            phase_head, mod_head = torch.chunk(head_emb, 2, dim = 2)
            rel = self.rel_proj_layer_norm(F.relu(self.rel_proj_layer(rel)))
            phase_relation, mod_relation, bias_relation = torch.chunk(rel, 3, dim = 2)
            phase_tail, mod_tail = torch.chunk(tail_emb, 2, dim = 2)

            phase_head = phase_head / (self.embedding_range.item() / self.pi)
            phase_relation = phase_relation / (self.embedding_range.item() / self.pi)
            phase_tail = phase_tail / (self.embedding_range.item() / self.pi)

            phase_score = (phase_head + phase_relation) - phase_tail

            mod_relation = torch.abs(mod_relation)
            bias_relation = torch.clamp(bias_relation, max=1)
            indicator = (bias_relation < -mod_relation)
            bias_relation[indicator] = -mod_relation[indicator]

            r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)

            phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight
            r_score = torch.norm(r_score, dim = 2) * self.modulus_weight

            # Adding hierarchy term in score
            if 0.0 < self.hierarchy_weight:
                h_scores = []
                for batch_idx in range(tail_hierarchy_level.size(0)): # iterating over each batch
                    a = head_hierarchy_level[batch_idx].expand(tail_hierarchy_level[batch_idx].shape) - tail_hierarchy_level[batch_idx]
                    b = r_direction[batch_idx] * a
                    c = self.hierarchy_weight * b
                    c = c.squeeze(1)
                    # creating new batch dim to concatenate along
                    c = c.unsqueeze(0)
                    h_scores.append(c)
                h_score = torch.cat(h_scores)
                return self.gamma.item() - (phase_score + r_score + h_score)
            
            return self.gamma.item() - (phase_score + r_score)

        
        elif ScoreFunction.DISTMULT.value == self.score_function:
            score = (head_emb * rel) * tail_emb
            return score.sum(dim = 2)
