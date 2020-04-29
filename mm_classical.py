##########################################
# Markovian Marginal : 1D Classical case #
# Isaac H. Kim 4/28/2020                 #
# MIT License                            #
##########################################
import numpy as np
import copy


class Marginals:
    """
    Attrs:
        vars(list): List of binary random variables.
        pdf(np.array): Joint probability distribution
    """
    def __init__(self, var_list):
        self.var_list = var_list
        self.n = len(var_list)
        self.pdf = np.ones([2] * self.n) / (1<<self.n)

    def remove_idx(self, k):
        """
        Remove the k'th variable in the list.

        Args:
            k(int): Index of the variable we are removing
        """
        # Remove the bit
        self.var_list.pop(k)
        # Sum
        self.pdf = self.pdf.sum(k)

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

        Returns:
            Marginal: Marginal of self to my_vars
        """
        mycopy = copy.deepcopy(self)
        vars_remove = list(set(self.var_list) - set(my_vars))
        mycopy.remove_vars(vars_remove)
        return mycopy
        
    def is_consistent_with(self, other, tolerance= 0.00001):
        """
        Checks if self is consistent with the other.

        Args:
            other(Marginal): Other marginal
            tolerance(float): error tolerance

        Returns:
            bool: True if the two are consistent with the tolerance
                  False otherwise
        """
        overlap = list(set(self.var_list).intersection(other.vars_list))
        marginal_self = self.marginal(overlap)
        marginal_other = other.marginal(overlap)
        if marginal_self.vars_list == marginal_other.vars_list:
            if abs(marginal_self.pdf - marginal_other.pdf).sum() > tolerance:
                return True
        return False

    def consistency_with(self, other):
        """
        Total variational distance between self and other marginalized
        over their overlaps.

        Args:
            other(Marginal): Other marginal

        Returns:
            float: Total variational distance
        """
        overlap = list(set(self.var_list).intersection(other.vars_list))
        m_self = self.marginal(overlap)
        m_other = other.marginal(overlap)
        perm_self = [overlap.index(var) for var in m_self.vars_list]
        perm_other = [overlap.index(var) for var in m_other.vars_list]
        pdf_self = m_self.pdf.transpose(perm_self)
        pdf_other = m_other.pdf.transpose(perm_other)

        return abs(pdf_self-pdf_other).sum()
        
        
    
                
