##########################################
# Markovian Marginal : 1D Classical case #
# Isaac H. Kim 4/28/2020                 #
# MIT License                            #
##########################################
import numpy as np
import copy

N = 100

class Marginals:
    """
    Attrs:
        vars(list): List of binary random variables.
        pdf(np.array): Joint probability distribution
    """
    def __init__(self, var_list):
        self.var_list = var_list
        self.n = len(bits)
        self.pdf = np.ones(1<<self.n) / (1<<self.n)

    def remove_idx(self, k):
        """
        Remove the k'th variable in the list.

        Args:
            k(int): Index of the variable we are removing
        """
        # Remove the bit
        self.var_list.pop(k)
        # Sum
        bitstrings = [s for s in range(1<<self.n) if s&(1<<k)]
        self.pdf = [pdf[s]+pdf[s^(1<<k)] for s in bitstrings]

    def remove_idxs(self, ks):
        """
        Remove the variables corresponding to the indices
        
        Args:
            ks(list(int)): Indices of the variables we are removing
        """
        for k in ks:
            self.remove_var(k)

    def remove_var(self, var):
        """
        Remove the variable

        Args:
            var: Name of the variable we want to remove
        """
        idx = self.var_list(var)
        self.remove_idx(idx)

    def remove_vars(self, my_vars):
        """
        Remove the variables

        Args:
            my_vars(list): A list of variables we want to remove
        """
        for var in my_vars:
            self.remove_var(self, var)

    def marginal(self, my_vars):
        """
        Returns the marginal pdf over my_vars.

        Args:
            my_vars(list): A list of variables
        """
        mycopy = copy.deepcopy(self)
        vars_remove = list(set(self.var_list) - set(my_vars))
        mycopy.remove_vars(vars_remove)
        return mycopy
        
    def is_consistent_with(self, other, tolerance= 0.00001):
        """
        Checks if self is consistent with the other.
        """
        overlap = list(set(self.vars_list).intersection(other.vars_list))
        marginal_self = self.marginal(overlap)
        marginal_other = other.marginal(overlap)
        if marginal_self.vars_list == marginal_other.vars_list:
            if sum(abs(marginal_self.pdf - marginal_other.pdf)) > tolerance:
                return True
        return False

    def __isub__(self, bits):
        for bit in bits:
            if bit in self.bits:
                