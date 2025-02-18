import argparse
import torch.cuda as cuda


def parse_args():
    parser = argparse.ArgumentParser(description="Run Recommender Model.")

    # Base setting
    parser.add_argument('--dim', type=int, default=32, help='Dim of latent vectors.')
    parser.add_argument('--layers', nargs='?', default='[32,16]', help="Dim of mlp layers.")
    parser.add_argument('--num_neg', type=int, default=4, help='Number of negative items.')
    parser.add_argument('--path', nargs='?', default='Data/', help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ML', help='Choose a dataset.')
    parser.add_argument('--device', nargs='?', default='cpu' if cuda.is_available() else 'cpu',
                        help='Which device to run the model.')

    # Important setting module
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--a_lr', type=float, default=0.005, help='Learning rate of gradient extract model.')
    parser.add_argument('--std', type=float, default=0.01, help='std.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of global epochs.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--items_limit', type=int, default=20, help='Limit of items.')
    parser.add_argument('--clients_limit', type=int, default=3, help='number of malicious clients.')
    parser.add_argument('--target_items', type=int, default=20, help='Limit of items.')
    parser.add_argument('--top_k_rec', type=int, default=50, help='top-k recommendation list.')

    # Hyperparameter module
    parser.add_argument('--proxy_nums', type=int, default=10, help='hyper-number of proxy items.')
    parser.add_argument('--lamda_aug', type=int, default=50, help='hyper-number of noise neighbors.')
    parser.add_argument('--noise', type=float, default=0.03, help='fixed noise ratio.')
    parser.add_argument('--lamda_att', type=int, default=30, help='fixed number of local epochs.')
    parser.add_argument('--beta_a', type=float, default=1.5, help='attack level.')
    parser.add_argument('--lamda_fil', type=int, default=30, help='fix to maintain recommendation.')
    parser.add_argument('--beta_r', type=float, default=50., help='fix to maintain recommendation..')
    parser.add_argument('--launch', type=int, default=8, help='hyper-attack start epoch.')
    parser.add_argument('--pop_num', type=int, default=10, help='hyper-popular item number.')
    parser.add_argument('--lamda', type=float, default=10., help='hyper-lamda.')

    # Defense module
    parser.add_argument('--use_sparse', type=bool, default=False, help='whether use sparse-based defense method.')
    parser.add_argument('--use_clip', type=bool, default=False, help='whether use clip-based defense method.')
    parser.add_argument('--grad_limit', type=float, default=1., help='Limit of l2-norm of item gradients.')
    parser.add_argument('--item_proportion', type=float, default=0.1, help='the proportion of items select from bank.')

    return parser.parse_args()


args = parse_args()
