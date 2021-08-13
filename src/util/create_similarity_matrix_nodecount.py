"""
Created on Sun May 09 15:12:09 2021
"""
import numpy as np
import scipy.stats
import json

def create_similarity_matrix_nodecount(steiner_trees):
    with open('../util/api_weight_label_vec.json', 'r') as fp:
        data = json.load(fp)
        api_vec = np.array(data['api_vec'])

        weight_vector = []
        candidate_num = len(steiner_trees)
        candidate_vec = np.zeros((candidate_num, api_vec.shape[1]))
        Gram_matrix = np.zeros((candidate_num, candidate_num))
        for (i, steiner) in enumerate(steiner_trees):
            vec = np.zeros((len(steiner.nodes), api_vec.shape[1]))
            for (v, node) in enumerate(steiner.nodes):
                vec[v, :] = api_vec[node]

            candidate_vec[i] = np.sum(vec, axis=0)
            weight_vector.append(1 + steiner.node_count)

        a = np.exp(1 / np.array(weight_vector))
        compatibility_vector = a /(np.sum(a))
        candidate_vec /= np.linalg.norm(candidate_vec, axis=1, keepdims=True)
        Gram_matrix = np.dot(candidate_vec, candidate_vec.T)

        return compatibility_vector, Gram_matrix

