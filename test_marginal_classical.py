# Tests for Marginal class (classical)
import numpy as np
from marginal_classical import Marginal

# Checking that the marginals inherited from the same global state
# are consistent.
for i in range(100):
    list_all = [i for i in range(10)]
    m_rand = Marginal.rand(list_all)
    list1 = np.random.choice(list_all, 5, replace=False)
    list2 = np.random.choice(list_all, 5, replace=False)
    m1 = m_rand.marginal(list1)
    m2 = m_rand.marginal(list2)
    print(m1.consistency_with(m2))
    assert m1.is_consistent_with(m2)
