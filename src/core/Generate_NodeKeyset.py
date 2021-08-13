import json
import numpy as np
import networkx as nx
from core.apiWeightedGraph import apiGraph

class NodeKeyset:
    def __init__(self, graph, keywords, api_categories):
        self.graph = graph
        self.keywords = keywords
        self.api_categories = api_categories

    def calcKeySet(self, v, keywords, num_of_keywords):
        '''Calculate the binary string corresponding to the keywords contained in the given node v'''
        category_set = set(self.api_categories[v])

        keySet = 0
        for i in range(num_of_keywords):
            if keywords[i] in category_set:
                keySet |= (1 << i)

        return keySet

    def generate_nodes(self, keywords):
        """
        Choosing an initial node.
        """
        num_of_keywords =len(keywords)
        nodes_keyset = {}
        keywords_nodes = []
        for v in range(self.graph.dimension):
            category_set = set(self.api_categories[v])
            keySet = 0
            for i in range(num_of_keywords):
                if keywords[i] in category_set:
                    keySet |= (1 << i)

            if keySet > 0:
                nodes_keyset[v] = keySet
                keywords_nodes.append(v)
            else:
                nodes_keyset[v] = 0

        return keywords_nodes, nodes_keyset