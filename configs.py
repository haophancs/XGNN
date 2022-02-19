import argparse

import torch


def arg_parse():
    parser = argparse.ArgumentParser(description='GraphPool arguments.')

    parser.add_argument('--datadir', dest='datadir')
    parser.add_argument('--prefix', dest='prefix')
    parser.add_argument('--epochs', dest='epochs')
    parser.add_argument('--seed', dest='seed')
    parser.add_argument('--cuda', dest='cuda')
    parser.add_argument('--lr', dest='lr')
    parser.add_argument('--dropout', dest='dropout')
    parser.add_argument('--weight-decay', dest='weight_decay')
    parser.add_argument('--outputdir', dest='outputdir')
    parser.set_defaults(datadir='/XGNN/Mutagenicity',
                        prefix='Mutagenicity',
                        epochs=80,
                        seed=200,
                        cuda=torch.cuda.is_available(),
                        lr=0.001,
                        dropout=0.1,
                        weight_decay=5e-4)
    return parser.parse_args()