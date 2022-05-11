import numpy as np
    
def fdownsampling(input_ = None): 
    input_ = double(input_)
    m,n = input_.shape
    LR = np.zeros((m / 2,n / 2))
    ii = 1
    jj = 1
    for i in np.arange(1,m - 1+2,2).reshape(-1):
        for j in np.arange(1,n - 1+2,2).reshape(-1):
            LR[ii,jj] = (input_(i,j) + input_(i,j + 1) + input_(i + 1,j) + input_(i + 1,j + 1)) / 4
            if (jj == n / 2):
                ii = ii + 1
                jj = 1
            else:
                jj = jj + 1
    
    return LR
    
    return LR