# -*- coding: utf-8 -*-
import queue
import numpy as np
from src.core.ratioSteinerTree import STree

MAX_RATIO = 999


class RatioSteinerAlgorithm:
    def __init__(self, graph, api_categories):
        self.graph = graph
        self.numOfNodes = graph.dimension
        self.api_categories = api_categories
        self.trees_dict = {}

    def run(self, keywords):
        '''
        Find the minimum steiner tree that contains the keywords, and return a list of the minimum steiner trees
        graph: co-invocation mashup-API mtrix
        keywords: requirements
        '''
        num_of_keywords = len(keywords)
        que = queue.PriorityQueue()
        result = queue.PriorityQueue()

        all_in = (1 << num_of_keywords) - 1

        for v in range(self.numOfNodes):
            keySet = self.calcKeySet(v, keywords, num_of_keywords)

            if keySet > 0:
                tree = STree(v, keySet)
                que.put(tree)
                self.addTree(v, tree)

        min_ratio = np.iinfo(np.int32).max
        while not que.empty():
            tree = que.get()

            if tree.keySet == all_in:
                result.put(tree)
                min_ratio = tree.ratio if tree.ratio < min_ratio else min_ratio
                continue


            v = tree.root
            neighbors, weights = self.graph.neighbors(v)

            # Tree growth
            for (idx, u) in enumerate(neighbors):
                t = self.getTree(u, tree.keySet)
                u_ratio = MAX_RATIO if t is None \
                    else t.ratio
                if ((tree.node_count + 1) * (tree.weight + weights[idx])) < u_ratio:
                    newTree = tree.grow(u, self.calcKeySet(u, keywords, num_of_keywords), weights[idx])

                    tmp = self.getTree(u, newTree.keySet)
                    if ((tmp == None or tmp.ratio > newTree.ratio) and (len(set(newTree.nodes)) == len(newTree.nodes))):
                        que.put(newTree)
                        self.addTree(u, newTree)

                        if newTree.keySet == all_in:
                            min_ratio = newTree.ratio if newTree.ratio < min_ratio else min_ratio

            # Tree merging
            trees = self.trees_dict.get(v)

            newTrees = []
            for key in trees.keys():
                t = trees[key]
                if t.keySet & tree.keySet == 0:
                    union_keySet = t.keySet | tree.keySet
                    union_tree = self.getTree(v, union_keySet)
                    union_ratio = MAX_RATIO \
                        if union_tree is None \
                        else union_tree.ratio

                    if ((t.node_count + tree.node_count - 1) * (t.weight + tree.weight)) < union_ratio:
                        newTree = tree.merge(t)

                        if (len(newTree.nodes) == (len(t.nodes) + len(tree.nodes) - 1)):
                            que.put(newTree)
                            newTrees.append(newTree)

                            if newTree.keySet == all_in:
                                min_ratio = newTree.ratio if newTree.ratio < min_ratio else min_ratio

            for t in newTrees:
                self.addTree(v, t)

        result_trees = []
        while not result.empty():
            next_tree = result.get()
            result_trees.append(next_tree)
        return result_trees
    def calcKeySet(self, v, keywords, num_of_keywords):
        category_set = set(self.api_categories[v])

        keySet = 0
        for i in range(num_of_keywords):

            if keywords[i] in category_set:
                keySet |= (1 << i)

        return keySet

    def addTree(self, root, tree):
        trees = self.trees_dict.get(root)

        if trees is None:
            trees = {}
            self.trees_dict[root] = trees

        trees[tree.keySet] = tree
    def getTree(self, root, keySet):
        trees = self.trees_dict.get(root)

        if trees is None:
            return None
        return trees.get(keySet)
