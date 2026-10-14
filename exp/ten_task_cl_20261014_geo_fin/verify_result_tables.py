"""Known analytical CL matrix and frozen real initial prediction checks."""
from result_tables import metrics

matrix=[[j+(10-(k-j-1) if k>=j+1 else 0) for j in range(10)] for k in range(11)]
expected={'AP':10.,'ForgetRate':5.,'BWT':-5.,'FWT':0.}
assert metrics(matrix)==expected
try:
    metrics(matrix[:-1])
except ValueError:
    pass
else:
    raise AssertionError('Incomplete matrix accepted')
print('PASS: known acquisition/forgetting matrix yields AP=10, ForgetRate=5, BWT=-5, FWT=0; incomplete matrices rejected')
