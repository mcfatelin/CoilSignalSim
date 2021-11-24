##############################################
# Test
##############################################

import numpy as np



A               = np.linspace(0,9,10)

print(A)


def process(x, i):
    if i==0:
        return x[i]
    else:
        x[i]        = x[i] + process(x,i-1)
    return x[i]


process(A,9)
print(A)
