# -*- coding: utf-8 -*-
class STree():
    def __init__(self, root, keySet = None, weight = 0):
        self.keySet = keySet
        self.root = root
        self.weight = weight
        self.node_count = 1
        self.ratio = 0
        self.nodes = [root]

    
    def grow(self, v, keySet, weight):
        """
        tree grow
        :param v:  the root of new tree
        :param keySet: the keySet of v
        :param w: weight
        :return:
        """
        newTree = STree(v)
        newTree.weight = self.weight + weight
        newTree.keySet = self.keySet | keySet
        newTree.node_count = self.node_count + 1
        newTree.ratio=newTree.node_count * newTree.weight
        newTree.nodes.extend(self.nodes)

        return newTree
    def merge(self, tree):
        '''tree merging'''
        newTree = STree(self.root)

        newTree.keySet = self.keySet | tree.keySet
        newTree.weight = self.weight + tree.weight
        newTree.node_count = self.node_count + tree.node_count - 1
        newTree.ratio=newTree.node_count * newTree.weight
        newTree.nodes = list(set(self.nodes).union(tree.nodes))

        return newTree
    def __lt__(self, other):
        if self.ratio!= other.ratio:
            return self.ratio < other.ratio
        elif  self.node_count!= other.node_count:
            return self.node_count< other.node_count
        else: return self.weight < other.weight

    def get_api_names(self, api_names):
        names = []
        if self.nodes:
            for v in self.nodes:
                names.append(api_names[v])

            return names

    def get_api_category(self, api_categories=None):
        category_set = set()
        if api_categories is not None:
            for j in self.nodes:
                for category in api_categories[j]:
                    category_set.add(category)

        return list(category_set)

    def display(self, u, api_names=None, api_categries=None):
        apis = self.nodes
        category_set = set()
        if api_categries is not None:
            for j in self.nodes:
                for category in api_categries[j]:
                    category_set.add(category)

        if api_names is not None:
            apis = self.get_api_names(api_names)

        print('nodes:', apis, ', keySet:', self.keySet, 'nodes_count:', self.node_count, 'weight:', self.weight,'ratio:',self.ratio);




