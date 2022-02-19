import copy
import random

import matplotlib.pyplot as plt
import networkx as nx
import torch.optim as optim

from configs import arg_parse
from models.generator import Generator
from utils import *

args = arg_parse()
random.seed(args.seed)

adj_list, features_list, labels, idx_map, idx_train, idx_val, idx_test = load_split_data(
    path=args.datadir,
    dataset=args.prefix
)


candidate_set = ['C.4', 'N.5', 'O.2', 'F.1', 'I.7', 'Cl.7', 'Br.5']
g = Generator(candidate_set, c=0, start=0, features_list=features_list,
              labels=labels, max_num_nodes=args.max_num_nodes, model_path=args.model_path)
optimizer = optim.Adam(g.parameters(), lr=args.lr, betas=(args.b1, args.b2))


def train_generator(c=0,
                    initial_node=None,
                    max_nodes=5):
    g.c = c

    for i in range(args.max_gen_step):

        optimizer.zero_grad()
        G = copy.deepcopy(g.G)
        p_start, a_start, p_end, a_end, G = g.forward(G)

        Rt = g.calculate_reward(G)
        loss = g.calculate_loss(Rt, p_start, a_start, p_end, a_end, G)
        loss.backward()
        optimizer.step()

        if G['num_nodes'] > max_nodes:
            g.reset_graph()
        elif Rt > 0:
            g.G = G


def generate_graph(c=0, max_nodes=5):
    g.c = c
    g.reset_graph()

    for i in range(args.max_gen_step):
        G = copy.deepcopy(g.G)
        p_start, a_start, p_end, a_end, G = g.forward(G)
        Rt = g.calculate_reward(G)

        if G['num_nodes'] > max_nodes:
            return g.G
        elif Rt > 0:
            g.G = G

    return g.G


def display_graph(G):
    G_nx = nx.from_numpy_matrix(np.asmatrix(G['adj'][:G['num_nodes'], :G['num_nodes']].numpy()))
    # nx.draw_networkx(G_nx)

    layout = nx.spring_layout(G_nx)
    nx.draw(G_nx, layout)

    coloring = torch.argmax(G['feat'], 1)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for i in range(7):
        nx.draw_networkx_nodes(G_nx, pos=layout, nodelist=[x for x in G_nx.nodes() if coloring[x] == i],
                               node_color=colors[i])
        nx.draw_networkx_labels(G_nx, pos=layout,
                                labels={x: candidate_set[i].split('.')[0] for x in G_nx.nodes() if coloring[x] == i})
    nx.draw_networkx_edges(G_nx, pos=layout, width=list(nx.get_edge_attributes(G_nx, 'weight').values()))
    nx.draw_networkx_edge_labels(G_nx, pos=layout, edge_labels=nx.get_edge_attributes(G_nx, "weight"))

    plt.show()


for i in range(1, 10):
    g.reset_graph()
    train_generator(c=1, initial_node=0, max_nodes=i)
    to_display = generate_graph(c=1, max_nodes=i)
    display_graph(to_display)
    print(g.model(to_display['feat'], to_display['adj'], None))
