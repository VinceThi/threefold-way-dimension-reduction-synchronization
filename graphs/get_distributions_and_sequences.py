from collections import Counter
import numpy as np
import networkx as nx


def get_distribution(array):
    sequence = sorted(array, reverse=True)
    count = Counter(sequence)
    deg, cnt = zip(*count.items())
    return deg, cnt


def get_degree_distribution(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degreeCount = Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    return deg, cnt


def get_degree_sequence(G):
    return np.sum(nx.to_numpy_array(G), axis=1)
