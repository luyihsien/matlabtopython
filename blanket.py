import numpy as np
    
def blanket(A = None): 
    M = double(A)
    m = M.shape
    e = 101
    U = np.zeros((m(1),m(2),e))
    B = np.zeros((m(1),m(2),e))
    for i in np.arange(1,m(1)+1,1).reshape(-1):
        for j in np.arange(1,m(2)+1,1).reshape(-1):
            U[i,j,1] = M(i,j)
            B[i,j,1] = M(i,j)
    
    for p in np.arange(2,e - 1+1,1).reshape(-1):
        for i in np.arange(1,m(1)+1,1).reshape(-1):
            for j in np.arange(1,m(2)+1,1).reshape(-1):
                if i == 1:
                    a = i
                else:
                    a = i - 1
                if j == 1:
                    b = j
                else:
                    b = j - 1
                if i == m(1):
                    c = i
                else:
                    c = i + 1
                if j == m(2):
                    d = j
                else:
                    d = j + 1
                U[i,j,p] = np.amax(U(i,j,p - 1) + 1,np.amax(U(i,j,p - 1),np.amax(np.amax(U(a,j,p - 1),U(i,b,p - 1)),np.amax(U(c,j,p - 1),U(i,d,p - 1)))))
                B[i,j,p] = np.amin(B(i,j,p - 1) - 1,np.amin(B(i,j,p - 1),np.amin(np.amin(B(a,j,p - 1),B(i,b,p - 1)),np.amin(B(c,j,p - 1),B(i,d,p - 1)))))
    
    V = np.zeros((51,1))
    V1 = np.zeros((m(1),m(2)))
    for p in np.arange(50,e - 1+1,1).reshape(-1):
        for i in np.arange(1,m(1)+1,1).reshape(-1):
            for j in np.arange(1,m(2)+1,1).reshape(-1):
                V1[i,j] = U(i,j,p) - B(i,j,p)
        V[p - 49,1] = sum(sum(V1))
    
    A = np.zeros((51,1))
    for p in np.arange(50,e - 1+1,1).reshape(-1):
        A[p - 49,1] = V(p - 49,1) / (2 * p)
    
    X = np.zeros((3,1))
    Y = np.zeros((3,1))
    kk = 1
    fd = 0
    for p in np.arange(50,e - 1+1,1).reshape(-1):
        X[kk,1] = np.log(p)
        Y[kk,1] = np.log(A(p - 49,1))
        if kk == 3:
            p1 = polyfit(X,Y,1)
            fdtemp = 2 - p1(1)
            kk = 0
            if p == 52:
                fdt1 = fdtemp
            fd = fd + fdtemp - fdt1
        kk = kk + 1
    
    X = np.zeros((51,1))
    Y = np.zeros((51,1))
    for p in np.arange(50,e - 1+1,1).reshape(-1):
        X[p - 49,1] = np.log(p)
        Y[p - 49,1] = np.log(A(p - 49,1))
    
    p = polyfit(X,Y,1)
    fd = 2 - p(1)
    return fd