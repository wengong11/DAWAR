# -*- coding: utf-8 -*-
import numpy as np
import json

class apiGraph:
    def __init__(self, matrix):
        self.matrix = matrix
        self.dimension = len(self.matrix)
        
    def neighbors(self, u):
        """
        :param u:
        :return:
        """
        neighbors = []
        weights = []
        for i in range(self.dimension):
            if self.matrix[u][i] > 0:
                neighbors.append(i)
                weights.append(self.matrix[u][i])
        return neighbors, weights


