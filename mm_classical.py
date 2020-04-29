###############################
# Markovian Marginal          #
# Isaac H. Kim 4/29/2020      #
# MIT License                 #
###############################
import numpy as np
from marginal_classical import Marginal


class MarkovMarginal(Marginal):
    """
    Marginal, but with an added attribute that encodes
    the internal Markov chain structure
    """
    def __init__(self, var_list):
        self.var_list = var_list
        self.n = len(var_list)
        self.pdf = np.ones([2] * self.n) / (1<<self.n)
        self.conditional_independence = []

    @classmethod
    def rand(cls, var_list):
        out = MarkovMarginal(var_list)
        for x in np.ndindex(tuple([2] * out.n)):
            out.pdf[x] = np.random.rand()
        out.pdf /= out.pdf.sum()
        return out

    def cmi(self, ab, bc):
        """
        Conditional mutual information
        H(AB) + H(BC) - H(B) - H(ABC)
        
        Args:
            ab, bc(list): Set AB and BC

        Returns:
            float: conditional mutual information
        """
        abc = list(set(ab)|set(bc))
        b = list(set(ab).intersection(set(bc)))
        s_abc = self.marginal(abc).entropy()
        s_ab = self.marginal(ab).entropy()
        s_bc = self.marginal(bc).entropy()
        s_b = self.marginal(b).entropy()
        return s_ab + s_bc - s_b - s_abc

    def cmi_constraints(self):
        """
        Sum of conditional mutual information constraints

        Returns:
            float: Sum of conditional mutual information constraints
        """
        cost = 0.0
        for c in self.conditional_independence:
            cost += self.cmi(c[0], c[1])
        return cost

        
    
    
