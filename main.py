import torch
import random
import numpy as np
from time import time
import torch.optim as optim
from scipy.stats import multivariate_normal
from torch import nn
from torch.nn import init

from gmm import CustomGaussianMixture
from parse import args
from data import load_dataset
from client import FedRecClient
from server import FedRecServer
from attack import malicious_client_by_random
from tqdm import tqdm

import torch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(result):
    with open(result, 'w') as f:

        args_str = ",".join([("%s=%s" % (k, v)) for k, v in args.__dict__.items()])
        print("Arguments: %s " % args_str)
        f.write(args_str + '\n')

        t0 = time()
        m_item, all_train_ind, all_test_ind, items_popularity = load_dataset(args.path + args.dataset)

        np.save(args.dataset + '_items_popularity.npy', items_popularity)

        _, target_items = torch.Tensor(-items_popularity).topk(args.target_items)
        target_items = target_items.tolist()

        server = FedRecServer(m_item, args.dim, eval(args.layers)).to(args.device)
        clients = []

        for train_ind, test_ind in tqdm(zip(all_train_ind, all_test_ind)):
            clients.append(
                FedRecClient(train_ind, test_ind, target_items, m_item, args.dim).to(args.device)
            )

        print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" %
              (time() - t0, len(clients), m_item,
               sum([len(i) for i in all_train_ind]),
               sum([len(i) for i in all_test_ind])))
        print("Target items: %s." % str(target_items))
        print("output format: (HR@20, NDCG@20, ER@50)")

        t1 = time()
        test_result_hr, test_result_undef, target_result = server.eval_(clients)  # evaluation
        print("Iteration 0(init), (%.7f) on test" % test_result_hr + "(%.7f) on test" % test_result_undef +
              ", (%.7f) on target." % tuple(target_result) +
              " [%.1fs]" % (time() - t1))

        items_emb_start_attack = []

        for epoch in range(1, args.launch + 1):
            t1 = time()
            rand_clients = np.arange(len(clients))
            np.random.shuffle(rand_clients)

            total_loss = []

            for i in range(0, len(rand_clients), args.batch_size):
                batch_clients_idx = rand_clients[i: i + args.batch_size]
                items_emb, loss = server.train_(clients, batch_clients_idx, None)
                total_loss.extend(loss)

            items_emb_start_attack.append(items_emb)
            total_loss = np.mean(total_loss).item()

            t2 = time()
            test_result_hr, test_result_undef, target_result = server.eval_(clients)
            training_info = "Iteration %d, loss = %.5f [%.1fs], (%.7f, %.7f) on test, (%.7f) on target. [%.1fs]" % (
                epoch, total_loss, t2 - t1,
                test_result_hr, test_result_undef,
                target_result[0],
                time() - t2
            )
            print(training_info)
            f.write(training_info + '\n')

        tmp_index = 0
        # malicious_clients_limit = max(int(len(clients) * args.clients_limit), 1)
        malicious_clients_limit = args.clients_limit

        for epoch in range(args.launch + 1, args.epochs + 1):

            if tmp_index == 0:
                clients.extend(
                    malicious_client_by_random(malicious_clients_limit, args.items_limit, m_item, args.dim,
                                               target_items,
                                               server.items_emb, args.proxy_nums, server))

            t1 = time()

            rand_clients = np.arange(len(clients))
            np.random.shuffle(rand_clients)

            total_loss = []

            for i in range(0, len(rand_clients), args.batch_size):
                batch_clients_idx = rand_clients[i: i + args.batch_size]
                _, loss = server.train_(clients, batch_clients_idx, items_emb_start_attack)
                total_loss.extend(loss)
            total_loss = np.mean(total_loss).item()

            t2 = time()
            test_result_hr, test_result_undef, target_result = server.eval_(clients)

            training_info = "Iteration %d, loss = %.5f [%.1fs], (%.7f, %.7f) on test, (%.7f) on target. [%.1fs]" % (
                epoch, total_loss, t2 - t1,
                test_result_hr, test_result_undef,
                target_result[0],
                time() - t2
            )
            print(training_info)
            f.write(training_info + '\n')
            tmp_index += 1


if __name__ == "__main__":
    setup_seed(20220110)
    result = 'Result/' + args.dataset + '_malicious_clients_' + str(
        args.clients_limit) + '_' + '_ProitMTA_target_items_{}.txt'.format(
        args.target_items)
    main(result)

    # for lamda in [1, 2, 5, 10, 20]:
    #     args.lamda = lamda
    #     print(f"With lamda={args.lamda}")
    #     setup_seed(20220110)
    #     result = 'results/hyper-lamda/' + args.dataset + '_lamda_' + str(
    #         args.lamda) + '_ProitMTA_target-items_{}.txt'.format(args.target_items)
    #     main(result)

    # for pop_num in [5, 10, 20, 50]:
    #     args.pop_num = pop_num
    #     print(f"With pop_num={args.pop_num}")
    #     setup_seed(20220110)
    #     result = 'results/hyper-popular/' + args.dataset + '_popular_' + str(
    #         args.pop_num) + '_ProitMTA_target-items_{}.txt'.format(args.target_items)
    #     main(result)

    # for aug in [10, 30, 50]:
    #     args.lamda_aug = aug
    #     print(f"With lamda_aug={args.lamda_aug}")
    #     setup_seed(20220110)
    #     result = 'results/hyper-aug/' + args.dataset + '_aug_' + str(
    #         args.lamda_aug) + '_ProitMTA_target-items_{}.txt'.format(args.target_items)
    #     main(result)

    # args.lamda_aug = 30
    # args.epoch = 30
    #
    # for launch in [4, 8, 12]:
    #     args.launch = launch
    #     print(f"With launch={args.launch}")
    #     setup_seed(20220110)
    #     result = 'results/hyper-launch/' + args.dataset + '_launch_' + str(
    #         args.launch) + '_ProitMTA_target-items_{}.txt'.format(args.target_items)
    #     main(result)
