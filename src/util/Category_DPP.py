import numpy as np
import math
import json

class DPP:
    def __init__(self):
        pass


    def run(self, kernel_matrix, max_iter, epsilon=1E-10):
        """
        fast implementation of the greedy algorithm
        :param kernel_matrix: 2-d array
        :param max_length: positive int
        :param epsilon: small positive scalar
        :return: list
        """

        c = np.zeros((max_iter, kernel_matrix.shape[0]))
        d = np.copy(np.diag(kernel_matrix))
        j = np.argmax(d)
        Yg = [j]
        iter = 0
        Z = list(range(kernel_matrix.shape[0]))
        while len(Yg) < max_iter:
            Z_Y = set(Z).difference(set(Yg))
            for i in Z_Y:
                if iter == 0:
                    ei = kernel_matrix[j, i] / np.sqrt(d[j])
                else:
                    ei = (kernel_matrix[j, i] - np.dot(c[:iter, j], c[:iter, i])) / np.sqrt(d[j])
                c[iter, i] = ei
                d[i] = d[i] - ei * ei
            d[j] = 0
            j = np.argmax(d)
            if d[j] < epsilon:
                break
            Yg.append(j)
            iter += 1

        return Yg
