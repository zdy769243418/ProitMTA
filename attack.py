import math
import os
import random

import torch
import torch.nn as nn

from gmm import CustomGaussianMixture
from parse import args
import numpy as np

os.environ['OMP_NUM_THREADS'] = '1'

torch.autograd.set_detect_anomaly(True)
from torch.optim.lr_scheduler import StepLR

import torch
import torch.nn as nn


class AttentionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# 定义 characteristic encoder
class CharacteristicEncoder(nn.Module):
    def __init__(self, item_embedding_dim, hidden_dim):
        super(CharacteristicEncoder, self).__init__()
        self.attention_net = AttentionNetwork(item_embedding_dim, hidden_dim)

    def forward(self, item_embeddings):
        attention_scores = self.attention_net(item_embeddings)
        transferable_characteristic_embedding = torch.sum(attention_scores * item_embeddings, dim=0)
        return transferable_characteristic_embedding

    def predict(self, item_embeddings):

        attention_scores = self.attention_net(item_embeddings)
        transferable_characteristic_embedding = torch.sum(attention_scores * item_embeddings, dim=0)
        return transferable_characteristic_embedding


import torch
import torch.nn as nn


class MetaNetwork(nn.Module):
    def __init__(self, characteristic_embedding_dim, hidden_dim):
        super(MetaNetwork, self).__init__()
        self.fc1 = nn.Linear(characteristic_embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, characteristic_embedding_dim * characteristic_embedding_dim)

    def forward(self, characteristic_embedding, proxy_embedding):
        x = torch.relu(self.fc1(characteristic_embedding))
        x = self.fc2(x)
        transformed_proxy_embedding = torch.matmul(proxy_embedding, x.view(args.dim, args.dim))
        return transformed_proxy_embedding

    def predict(self, characteristic_embedding, proxy_embedding):
        x = torch.relu(self.fc1(characteristic_embedding))
        x = self.fc2(x)
        transformed_proxy_embedding = torch.matmul(proxy_embedding, x.view(args.dim, args.dim))
        return transformed_proxy_embedding

    def get_strategy(self, characteristic_embedding):
        x = torch.relu(self.fc1(characteristic_embedding))
        x = self.fc2(x)
        return x.view(args.dim, args.dim)


class OurAttackClient(nn.Module):
    def __init__(self, target_items, clustered_items, item_limits, m_item, dim, item_embeddings, proxy_nums, server):
        super().__init__()
        self._target_ = target_items
        self.m_item = m_item
        self.dim = dim
        self._user_emb = nn.Embedding(1, self.dim)
        self._clustered_items = clustered_items
        self._item_limits = item_limits
        self._assigned_items = self.get_assigned_items(clustered_items, item_limits)
        self.characteristic_encoder = CharacteristicEncoder(args.dim, args.dim * 2)
        self.meta_network = MetaNetwork(args.dim, args.dim * 2)
        self.item_embeddings = item_embeddings
        self.proxy_nums = proxy_nums

        self.pre_item_embs = server.items_emb.weight.clone().detach()

        self.optimizer = torch.optim.Adam(
            list(self.meta_network.parameters()) + list(self.characteristic_encoder.parameters()), lr=args.a_lr)
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.85)
        self.gmm = CustomGaussianMixture(n_components=args.proxy_nums)

    def forward(self, user_emb, items_emb, linear_layers, for_train=False):
        user_emb = user_emb.repeat(len(items_emb), 1)
        v = torch.cat((user_emb, items_emb), dim=-1)
        for i, (w, b) in enumerate(linear_layers):
            v = v @ w.t() + b
            if i < len(linear_layers) - 1:
                v = v.relu()
            else:
                v = v.sigmoid()
        return v.view(-1)

    def train_on_user_emb(self, user_emb, items_emb, linear_layers):
        predictions = self.forward(user_emb.requires_grad_(False), items_emb, linear_layers, for_train=True)
        pos_scores = predictions[0]
        neg_scores = predictions[1:]
        loss = torch.sum(neg_scores - pos_scores)
        loss = nn.Sigmoid()(loss)
        return loss

    def train_(self, item_embeddings, items_emb, linear_layers, items_emb_start_attack):

        transform_functions = []

        all_items_emb = items_emb.clone().detach().requires_grad_(False)
        batch_items_emb_grad = torch.zeros_like(items_emb)
        batch_linear_layers_grad = [[torch.zeros_like(w), torch.zeros_like(b)] for (w, b) in linear_layers]
        items_emb = items_emb.clone().detach().requires_grad_(True)

        target_items = self._target_

        items_emb_0 = items_emb_start_attack[0]
        items_emb_8 = items_emb_start_attack[len(items_emb_start_attack) - 1]

        k = args.pop_num
        accumulated_gradient = items_emb_8 - items_emb_0
        similarities = torch.nn.functional.cosine_similarity(items_emb_8, accumulated_gradient, dim=1)
        _, top_k_indices = similarities.topk(k, largest=True)
        np.save(args.dataset + '_top_k_indices.npy', np.array(top_k_indices))

        top_k_items_emb = all_items_emb[top_k_indices]

        v_t = torch.mean(top_k_items_emb, dim=0)

        proxy_items_embeddings, cluster_item_embeddings, target_item_probs = self.gmm_train(item_embeddings,
                                                                                            target_items)
        for s in range(args.lamda_aug):
            loss = 0.0
            for i in range(self.proxy_nums):
                self.optimizer.zero_grad()

                item_embeddings = cluster_item_embeddings[i]
                if s != i:
                    item_embeddings = item_embeddings + torch.randn_like(item_embeddings) * args.noise
                characteristic_embedding = self.characteristic_encoder(item_embeddings)

                transformed_proxy_embedding = self.meta_network(characteristic_embedding.float(),
                                                                proxy_items_embeddings[i].float())
                transformed_proxy_embedding = transformed_proxy_embedding.view(1, -1)
                loss += torch.norm(transformed_proxy_embedding - v_t * args.lamda_att, p=2)

            loss.backward()
            self.optimizer.step()

        transformed_proxy_embeddings = []

        for i in range(self.proxy_nums):
            item_embeddings = cluster_item_embeddings[i]
            proxy_items_embedding = proxy_items_embeddings[i]
            characteristic_embedding = self.characteristic_encoder.predict(item_embeddings)
            transformed_proxy_embeddings.append(self.meta_network.predict(characteristic_embedding.float(),
                                                                          proxy_items_embedding.float()))

        index = 0
        for target_item in target_items:
            target_item_feature = items_emb[target_item].clone().detach().requires_grad_(False)
            transformed_target_item = torch.zeros_like(target_item_feature)
            for i in range(self.proxy_nums):
                transformed_target_item += transformed_proxy_embeddings[i] * target_item_probs[index][i]

            target_item_gradient = (target_item_feature - transformed_target_item) * args.beta_a

            batch_items_emb_grad[target_item] = target_item_gradient
            index += 1

            # items_emb_grad_norm = target_item_gradient.norm(2, dim=-1, keepdim=True)
            #
            # print(items_emb_grad_norm)

        for i in range(self.proxy_nums):
            item_embeddings = cluster_item_embeddings[i]
            characteristic_embedding = self.characteristic_encoder.predict(item_embeddings)
            transform_function = self.meta_network.get_strategy(characteristic_embedding)
            transform_functions.append(transform_function.detach().numpy())
        self.gmm.transform_functions = transform_functions

        k = args.lamda_fil
        distances = torch.norm(self.pre_item_embs - all_items_emb, dim=1)
        top_k_indices = torch.argsort(distances, descending=True)[:k].tolist()
        for item_index in top_k_indices:
            if item_index in self._target_:
                top_k_indices.remove(item_index)

        real_train_items = self._target_ + top_k_indices

        g_i_epoch = (all_items_emb - self.pre_item_embs[-1])

        self.pre_item_embs = all_items_emb

        batch_items_emb_grad[top_k_indices] = g_i_epoch[top_k_indices] * args.beta_r

        return real_train_items, batch_items_emb_grad[real_train_items], batch_linear_layers_grad, 0.0

    def get_similar_items(self, target_item, all_item_emb):
        distance = torch.nn.functional.pairwise_distance(
            all_item_emb[target_item].clone().detach().repeat(len(all_item_emb), 1),
            all_item_emb.clone().detach())
        indexes = torch.argsort(distance, descending=False).tolist()[:args.top_k]
        if target_item in indexes:
            indexes.remove(target_item)
        return indexes

    def eval_(self, _items_emb, _linear_layers):
        return None, None

    def get_assigned_items(self, clustered_items, item_limits):
        if len(clustered_items) > item_limits:
            assigned_items = np.random.choice(clustered_items, item_limits, replace=False).tolist()
        else:
            assigned_items = clustered_items
        return assigned_items

    def construct_data(self, assigned_items):
        items, labels = [], []
        for pos in assigned_items:
            items.append(pos)
            labels.append(1.)
            if pos in self._target_:
                continue

        negs = [i for i in range(self.m_item) if i not in items and i not in self._clustered_items]
        neg_count = 0
        pos_count = len(items)
        for neg in negs:
            items.append(neg)
            labels.append(0.)
            neg_count += 1
            if neg_count >= 4 * pos_count: break
        return items, labels

    def gmm_train(self, item_embeddings, target_item_indices):

        target_item_embeddings = item_embeddings(torch.tensor(target_item_indices))
        all_item_indices = torch.arange(item_embeddings.num_embeddings)
        remaining_item_indices = all_item_indices[~torch.isin(all_item_indices, torch.tensor(target_item_indices))]
        remaining_item_embeddings = item_embeddings(remaining_item_indices)

        self.gmm.fit(remaining_item_embeddings.detach().numpy())

        cluster_centers = torch.tensor(self.gmm.means_)
        cluster_item_embeddings = []
        for i in range(args.proxy_nums):
            cluster_mask = self.gmm.predict(remaining_item_embeddings.detach().numpy()) == i
            cluster_item_embeddings.append(remaining_item_embeddings.detach()[cluster_mask])
        target_item_probs = self.gmm.predict_proba(target_item_embeddings.detach().numpy())

        return cluster_centers, cluster_item_embeddings, torch.from_numpy(target_item_probs)


def malicious_client_by_random(malicious_clients_limit, items_limit, m_item, dim, target_items, items_emb, proxy_nums,
                               server):
    clients = []
    for i in range(malicious_clients_limit):
        items = np.random.choice([ii for ii in range(m_item)], items_limit, replace=False).tolist()
        clients.append(
            OurAttackClient(target_items, items, items_limit, m_item, dim, items_emb, proxy_nums, server).to(
                args.device))
    return clients
