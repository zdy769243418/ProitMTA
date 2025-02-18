import torch
import torch.nn as nn

from gmm import CustomGaussianMixture
from parse import args


class FedRecServer(nn.Module):
    def __init__(self, m_item, dim, layers):
        super().__init__()
        self.m_item = m_item
        self.dim = dim
        self.layers = layers

        self.items_emb = nn.Embedding(m_item, dim)
        nn.init.normal_(self.items_emb.weight, std=args.std)

        layers_dim = [2 * dim] + layers + [1]
        self.linear_layers = nn.ModuleList([nn.Linear(layers_dim[i - 1], layers_dim[i])
                                            for i in range(1, len(layers_dim))])
        for layer in self.linear_layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
        if args.use_sparse:
            self.item_grad_bank = torch.zeros_like(self.items_emb.weight).to(args.device)
            self.item_grad_factor = torch.ones_like(self.items_emb.weight).to(args.device)

    def train_(self, clients, batch_clients_idx, items_emb_start_attack):
        items_emb = self.items_emb.weight
        linear_layers = [[layer.weight, layer.bias] for layer in self.linear_layers]
        batch_loss = []
        batch_items_emb_grad = torch.zeros_like(items_emb)
        batch_linear_layers_grad = [[torch.zeros_like(w), torch.zeros_like(b)] for (w, b) in linear_layers]

        for idx in batch_clients_idx:
            client = clients[idx]

            items, items_emb_grad, linear_layers_grad, loss = client.train_(self.items_emb, items_emb, linear_layers,
                                                                            items_emb_start_attack)

            with torch.no_grad():
                if args.use_clip:
                    items_emb_grad_norm = items_emb_grad.norm(2, dim=-1, keepdim=True)
                    too_large = items_emb_grad_norm[:, 0] > args.grad_limit
                    items_emb_grad[too_large] /= (items_emb_grad_norm[too_large] / args.grad_limit)

                batch_items_emb_grad[items] += items_emb_grad

                for i in range(len(linear_layers)):
                    batch_linear_layers_grad[i][0] += linear_layers_grad[i][0]
                    batch_linear_layers_grad[i][1] += linear_layers_grad[i][1]

            if loss is not None:
                batch_loss.append(loss)

        if args.use_sparse:
            self.item_grad_bank += batch_items_emb_grad
            tmp_item_grad_bank_norm = self.item_grad_bank.norm(2, dim=-1, keepdim=False)
            _, selected = torch.topk(tmp_item_grad_bank_norm, k=int(args.item_proportion * len(items_emb)))

            batch_items_emb_grad = torch.zeros_like(items_emb)
            batch_items_emb_grad[selected] += self.item_grad_bank[selected]

            tmp_grad_limit = torch.sum(tmp_item_grad_bank_norm[selected]) / len(selected)
            batch_items_emb_grad_norm = batch_items_emb_grad.norm(2, dim=-1, keepdim=True)
            batch_too_large = batch_items_emb_grad_norm[:, 0] > tmp_grad_limit
            batch_items_emb_grad[batch_too_large] /= (batch_items_emb_grad_norm[batch_too_large] / tmp_grad_limit)

            self.item_grad_bank[selected] -= self.item_grad_bank[selected]
        with torch.no_grad():
            self.items_emb.weight.data.add_(batch_items_emb_grad, alpha=-args.lr)
            for i in range(len(linear_layers)):
                self.linear_layers[i].weight.data.add_(batch_linear_layers_grad[i][0], alpha=-args.lr)
                self.linear_layers[i].bias.data.add_(batch_linear_layers_grad[i][1], alpha=-args.lr)
        return self.items_emb.weight.clone().detach(), batch_loss

    def clip_grad(self, l2_norm_clip, record):
        try:
            l2_norm = torch.norm(record)
        except:
            l2_norm = record.l2estimate()
        if l2_norm < l2_norm_clip:
            return record
        else:
            return record / float(torch.abs(torch.tensor(l2_norm) / l2_norm_clip))

    def eval_(self, clients):
        items_emb = self.items_emb.weight
        linear_layers = [(layer.weight, layer.bias) for layer in self.linear_layers]
        test_cnt, test_results_hr, test_results_undef = 0, 0., 0.
        target_cnt, target_results = 0, 0.

        with torch.no_grad():
            for client in clients:
                test_result, target_result = client.eval_(items_emb, linear_layers)
                if test_result is not None:
                    test_cnt += 1
                    test_results_hr += test_result[0]
                    test_results_undef += test_result[1]
                if target_result is not None:
                    target_cnt += 1
                    target_results += target_result
        return test_results_hr / test_cnt, test_results_undef / test_cnt, target_results / target_cnt
