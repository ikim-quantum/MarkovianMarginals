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
    def __init__(self):
        self.conditional_independence = []

    def cmi(self, ab, bc):
        """
        Conditional mutual information
        H(AB) + H(BC) - H(B) - H(ABC)
        
        Args:
            ab, bc(list): Set AB and BC

        Returns:
            float: conditional mutual information
        """
        abc = list(set(ab)+set(bc))
        b = list(set(ab).intersection(set(bc)))
        s_abc = self.marginal(abc).pdf
        s_ab = self.marginal(ab).pdf
        s_bc = self.marginal(bc).pdf
        s_b = self.marginal(b).pdf
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

        
    
    
