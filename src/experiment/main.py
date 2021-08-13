import numpy as np
import time
import json
from core.apiRatioGraph import apiGraph
from algorithm.div_ratioSteiner_for_DPP import RatioSteinerAlgorithm
from util.create_similarity_matrix_nodecount import create_similarity_matrix_nodecount
from util.create_similarity_matrix_weight import create_similarity_matrix_weight
from util.create_similarity_matrix_ratio import create_similarity_matrix_ratio
from util.Category_DPP import DPP
from itertools import combinations

def load_data():
    """
    read processed data from raw data (mashup.txt)
    :return:
    """
    with open('../../dataset/processed_data.json', 'r') as fp:
        data = json.load(fp)
        graph = apiGraph(data['matrix'])
        api_categories = data['api_categories']
        api_names = data['api_names']

        return graph, api_categories, api_names
def merge_api_categories(api_categories):
    """
    :param api_categories:
    :return:
    """
    category_set = set()
    for c in api_categories:
        category_set = category_set.union(c)

    return np.array(list(category_set))
def random_generate_keywords(categories, kyeword_num, batches, seed = 1):
    """
    Randomly generate a specified number of keywords
    :param categories:
    :param kyeword_num:
    :param batches:
    :param seed:
    :return:
    """
    np.random.seed(seed)
    keywords = []
    category_num = len(categories)

    for i in range(batches):
        indices = np.random.choice(category_num, kyeword_num,replace=False)
        keywords.append(categories[indices])

    return keywords
def generate_keywords_from_mashup(keyword_num, batches, seed = 1):
    """read data from mymashup_for_test.json"""
    np.random.seed(seed)

    test_data = json.load(open('../../dataset/mymashup_for_test.json', 'r'))
    test_apis = test_data['test_apis']
    test_categories = test_data['test_categories']

    category_num = len(test_categories[str(keyword_num)])
    apis = np.array(test_apis[str(keyword_num)])
    api_categories = np.array(test_categories[str(keyword_num)])

    indices = np.random.choice(category_num, batches, replace=False)

    return api_categories[indices].tolist(), apis[indices].tolist()
def mashup_keywords(times, tree_num, seed):
    """
    test mashup keywords(MK)
    :param times:
    :return:
    """
    nodes, apis, categories, weights, nodecount, costs, costs1, costs2, test_apis = query_for_three_methods(generate_keywords_from_mashup, times, tree_num, False, seed)
    with open('../../output/DPP/results_div_embedding_ratiosteiners_mashup_keywords.json', 'w') as fp:
        json.dump({'nodes': nodes.tolist(), 'apis': apis.tolist(), 'categories': categories.tolist(), 'weights': weights.tolist(), 'node_count': nodecount.tolist(), 'costs': costs.tolist(), 'costs_stenier': costs1.tolist(), 'costs_DPP': costs2.tolist(), 'test_apis': test_apis.tolist()}, fp)
def random_keywords(times, tree_num, seed):
    """
    test random keywords(RK)
    :param times:
    :return:
    """
    nodes, apis, categories, weights, nodecount, costs, costs1, costs2, test_apis = query_for_three_methods(random_generate_keywords, times, tree_num, True, seed)

    with open('../../output/DPP/results_div_embedding_ratioteiners_random_keywords.json', 'w') as fp:
        json.dump({'nodes': nodes.tolist(), 'apis': apis.tolist(), 'categories': categories.tolist(), 'weights': weights.tolist(), 'node_count': nodecount.tolist(), 'costs': costs.tolist(), 'costs_stenier': costs1.tolist(), 'costs_DPP': costs2.tolist(), 'test_apis': test_apis.tolist()}, fp)
def query_for_three_methods(generate_keywords_func, times, tree_num, is_random, seed):
    """
    We will determine our algorithm based on the scale of three cases: DPP(nodecount)、DPP(weight) and DPP(ratio)
    :param generate_keywords_func:randomly generately query keywords
    :param times:run times
    :param seed: random seed
    :return:
    """
    # load data
    graph, api_categories, api_names = load_data()
    random_categories = merge_api_categories(api_categories)

    nodes = np.empty((3, 5, times, tree_num), object)
    nodecount = np.zeros((3, 5, times, tree_num))
    costs = np.zeros((3, 5, times))
    costs1 = np.zeros((3, 5, times))
    costs2 = np.zeros((3, 5, times))
    weights = np.zeros((3, 5, times, tree_num))
    apis = np.empty((3, 5, times, tree_num), object)
    categories = np.empty((3, 5, times, tree_num), object)
    test_apis = np.empty((5, times), object)

    keywords_num_options = [2, 3, 4, 5, 6]

    for (idx, keywords_num) in enumerate(keywords_num_options):
        seed = seed + 10
        if is_random:
            keywords_batch = generate_keywords_func(random_categories, keywords_num, times, seed=seed)
        else:
            keywords_batch, test_apis[idx, :] = generate_keywords_func(keywords_num, times, seed=seed)

        for (batch, keywords) in enumerate(keywords_batch):
            print(keywords)
            # ratioSteiner + DPP(nodecount)
            steiners_nodecount, DPP_nodecount1, cost_nodecount1, cost_run1, cost_DPP1 = run_ratio_steiner_nodecount(graph, keywords, api_categories, api_names, tree_num, test_apis)

            if len(DPP_nodecount1):
                for (idx_tree, v) in enumerate(DPP_nodecount1):
                    nodes[0, idx, batch, idx_tree], apis[0, idx, batch, idx_tree], categories[0, idx, batch, idx_tree], weights[0, idx, batch, idx_tree], nodecount[0, idx, batch, idx_tree], costs[0, idx, batch], costs1[0, idx, batch], costs2[0, idx, batch],  = \
                        steiners_nodecount[v].nodes, steiners_nodecount[v].get_api_names(api_names), steiners_nodecount[v].get_api_category(
                            api_categories), steiners_nodecount[v].weight, steiners_nodecount[v].node_count, cost_nodecount1, cost_run1, cost_DPP1
            else:
                print('ratio steiner: search ', keywords, ' failed!')
                for idx_tree in range(tree_num):
                    nodes[0, idx, batch, idx_tree], apis[0, idx, batch, idx_tree], categories[0, idx, batch, idx_tree], weights[0, idx, batch, idx_tree], nodecount[0, idx, batch, idx_tree], costs[0, idx, batch], costs1[0, idx, batch], costs2[0, idx, batch], \
                    = [], [], [], 0, 0, cost_nodecount1, cost_run1, cost_DPP1

            # ratioSteiner + DPP(weight)
            steiners_weight, DPP_weight1, cost_DPP2 = run_ratio_steiner_weight(steiners_nodecount, tree_num, api_names, test_apis)
            cost_weight = cost_run1 + cost_DPP2
            if len(DPP_weight1):
                for (idx_tree, v) in enumerate(DPP_weight1):
                    nodes[1, idx, batch, idx_tree], apis[1, idx, batch, idx_tree], categories[1, idx, batch, idx_tree], \
                    weights[1, idx, batch, idx_tree], nodecount[1, idx, batch, idx_tree], costs[1, idx, batch], costs1[1, idx, batch], costs2[1, idx, batch], \
                     = steiners_weight[v].nodes, steiners_weight[v].get_api_names(api_names), steiners_weight[v].get_api_category(
                            api_categories), steiners_weight[v].weight, steiners_weight[v].node_count, cost_weight, cost_run1, cost_DPP2
            else:
                print('ratio steiner: search ', keywords, ' failed!')
                for idx_tree in range(tree_num):
                    nodes[1, idx, batch, idx_tree], apis[1, idx, batch, idx_tree], categories[1, idx, batch, idx_tree], \
                    weights[1, idx, batch, idx_tree], nodecount[1, idx, batch, idx_tree],  costs[1, idx, batch], costs1[1, idx, batch], costs2[1, idx, batch] = [], [], [], 0, 0, cost_weight, cost_run1, cost_DPP2

            # ratioSteiner + DPP(ratio): our approach
            steiners_ratio3, DPP_ratio3, cost_DPP3 = run_ratio_steiner_ratio(steiners_nodecount, tree_num,
                                                                                  api_names, test_apis)
            cost_ratio = cost_run1 + cost_DPP3
            if len(DPP_ratio3):
                for (idx_tree, v) in enumerate(DPP_ratio3):
                    nodes[2, idx, batch, idx_tree], apis[2, idx, batch, idx_tree], categories[
                        2, idx, batch, idx_tree], \
                    weights[2, idx, batch, idx_tree], nodecount[2, idx, batch, idx_tree], costs[2, idx, batch], costs1[2, idx, batch], costs2[2, idx, batch], \
                     = steiners_ratio3[v].nodes, steiners_ratio3[
                        v].get_api_names(api_names), steiners_ratio3[v].get_api_category(
                        api_categories), steiners_ratio3[v].weight, steiners_ratio3[v].node_count, cost_ratio, cost_run1, cost_DPP3
            else:
                print('ratio steiner: search ', keywords, ' failed!')
                for idx_tree in range(tree_num):
                    nodes[2, idx, batch, idx_tree], apis[2, idx, batch, idx_tree], categories[
                        2, idx, batch, idx_tree], \
                    weights[2, idx, batch, idx_tree], nodecount[2, idx, batch, idx_tree], costs[2, idx, batch], costs1[2, idx, batch], costs2[2, idx, batch], \
                     = [], [], [], 0, 0, cost_ratio, cost_run1, cost_DPP3
            print('>', end='')
            if (batch + 1) % 10 == 0:
                print('')

        print("%d keywords tested!" % keywords_num)
    print("Successfully tested!")

    return nodes, apis, categories, weights, nodecount, costs, costs1, costs2, test_apis
def nodecount_for_DPP(steiner_trees):
    # construct similarity matrix
    weight_vector, similarity_matrix = create_similarity_matrix_nodecount(steiner_trees)
    # construct DPP kernel matrix
    kernel_matrix = weight_vector.reshape(
        (similarity_matrix.shape[0], 1)) * similarity_matrix * weight_vector.reshape(
        (1, similarity_matrix.shape[0]))

    return kernel_matrix, similarity_matrix
def weight_for_DPP(steiner_trees):
    # construct similarity matrix
    weight_vector, similarity_matrix = create_similarity_matrix_weight(steiner_trees)
    # construct DPP kernel matrix
    kernel_matrix = weight_vector.reshape(
        (similarity_matrix.shape[0], 1)) * similarity_matrix * weight_vector.reshape(
        (1, similarity_matrix.shape[0]))

    return kernel_matrix, similarity_matrix
def ratio_for_DPP(steiner_trees):
    # construct similarity matrix
    weight_vector, similarity_matrix = create_similarity_matrix_ratio(steiner_trees)
    # construct DPP kernel matrix
    kernel_matrix = weight_vector.reshape(
        (similarity_matrix.shape[0], 1)) * similarity_matrix * weight_vector.reshape(
        (1, similarity_matrix.shape[0]))

    return kernel_matrix, similarity_matrix
def adaptive_threshold(result_trees):
    for alist_i, alist_j in combinations(result_trees, 2):
        if set(alist_j.nodes) == set(alist_i.nodes):
            if alist_j in result_trees:
                result_trees.remove(alist_j)

    return result_trees
def metric(steiner_trees, DPP_results, similarity_matrix, api_names, test_apis):
    DPP_metric_object = DPP_metric(steiner_trees, DPP_results)
    ILAD, ILMD = DPP_metric_object.compare_similarity(similarity_matrix)

    return ILAD, ILMD
def run_ratio_steiner_nodecount(graph, keywords, api_categories, api_names, tree_num, test_apis):

    anchor = time.time()
    ratiosteiner = RatioSteinerAlgorithm(graph, api_categories)
    result_trees = ratiosteiner.run(keywords)
    anchor_ratio = time.time()
    cost_run = anchor_ratio - anchor

    # agjust parameter Z
    if result_trees and len(result_trees) > tree_num:
        steiner_trees = adaptive_threshold(result_trees)
        kernel_matrix, similarity_matrix = nodecount_for_DPP(steiner_trees)
        DPP_object = DPP()
        DPP_results = DPP_object.run(kernel_matrix, tree_num)
        cost_DPP = time.time() - anchor_ratio
        cost = time.time() - anchor

        return steiner_trees, DPP_results, cost, cost_run, cost_DPP

    return [], [], 0, 0, 0
def run_ratio_steiner_weight(steiner_trees, tree_num, api_names, test_apis):

    if steiner_trees:
        anchor = time.time()
        kernel_matrix, similarity_matrix = weight_for_DPP(steiner_trees)
        DPP_object = DPP()
        DPP_results = DPP_object.run(kernel_matrix, tree_num)
        cost_DPP = time.time() - anchor


        return steiner_trees, DPP_results, cost_DPP

    return [], [], 0
def run_ratio_steiner_ratio(steiner_trees, tree_num, api_names, test_apis):

    if steiner_trees:
        anchor = time.time()
        kernel_matrix, similarity_matrix = ratio_for_DPP(steiner_trees)
        DPP_object = DPP()
        DPP_results = DPP_object.run(kernel_matrix, tree_num)
        cost_DPP = time.time() - anchor

        return steiner_trees, DPP_results, cost_DPP

    return [], [], 0
def get_api_names(api_names, nodes):
    """
    将api的索引列表转换为api name列表
    :param api_names:
    :param nodes:
    :return:
    """
    names = []
    for v in nodes:
        names.append(api_names[v])
    return names

if __name__ == '__main__':
    mashup_keywords(times=100, tree_num=10, seed=1322)
    # random_keywords(times=100, tree_num=10, seed=1322)
