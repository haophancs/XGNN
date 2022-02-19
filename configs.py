import argparse

import torch


def arg_parse():
    parser = argparse.ArgumentParser(description='GraphPool arguments.')

    parser.add_argument('--datadir', dest='datadir')
    parser.add_argument('--prefix', dest='prefix')
    parser.add_argument('--epochs', dest='epochs', type=int)
    parser.add_argument('--seed', dest='seed', type=int)
    parser.add_argument('--cuda', dest='cuda', type=bool)
    parser.add_argument('--lr', dest='lr', type=float)
    parser.add_argument('--b1', dest='b1', type=float)
    parser.add_argument('--b2', dest='b2', type=float)
    parser.add_argument('--hyp1', dest='hyp1', type=float)
    parser.add_argument('--hyp2', dest='hyp2', type=float)
    parser.add_argument('--dropout', dest='dropout', type=float)
    parser.add_argument('--weight-decay', dest='weight_decay', type=float)
    parser.add_argument('--max-num-nodes', dest='max_num_nodes', type=int)
    parser.add_argument('--rollout', dest='rollout', type=int)
    parser.add_argument('--max-gen-step', dest='max_gen_step', type=int)
    parser.add_argument('--model-path', dest='model_path')
    parser.set_defaults(datadir='./data/MUTAG/',
                        prefix='MUTAG_',
                        epochs=80,
                        seed=200,
                        cuda=torch.cuda.is_available(),
                        lr=0.001,
                        b1=0.9,
                        b2=0.99,
                        hyp1=1,
                        hyp2=2,
                        dropout=0.1,
                        weight_decay=5e-4,
                        max_num_nodes=28,
                        rollout=10,
                        max_gen_step=10,
                        model_path='./ckpt/MUTAG')
    return parser.parse_args()
