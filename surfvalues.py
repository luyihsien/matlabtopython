import numpy as np
    
def values(): 
    x = np.arange(0,4+1,1)
    y = np.arange(0,4+1,1)
    f = np.array([[4,3,6,5,4],[3,5,3,4,3],[5,3,4,3,5],[3,3,3,4,5],[4,3,4,5,3]])
    for i in np.arange(1,4+1).reshape(-1):
        for j in np.arange(1,5+1).reshape(-1):
            deltax[i,j] = (f(i + 1,j) - f(i,j)) / (x(i + 1) - x(i))
    
    for i in np.arange(1,5+1).reshape(-1):
        for j in np.arange(1,4+1).reshape(-1):
            deltay[i,j] = (f(i,j + 1) - f(i,j)) / (y(j + 1) - y(j))
    
    for j in np.arange(1,5+1).reshape(-1):
        dx[1,j] = deltax(1,j) - (deltax(2,j) - deltax(1,j)) / 2
    
    for i in np.arange(2,4+1).reshape(-1):
        for j in np.arange(1,5+1).reshape(-1):
            dx[i,j] = (deltax(i,j) + deltax(i - 1,j)) / 2
    
    for j in np.arange(1,5+1).reshape(-1):
        dx[5,j] = deltax(4,j) + (deltax(4,j) - deltax(4,j)) / 2
    
    for i in np.arange(1,5+1).reshape(-1):
        dy[i,1] = deltay(i,1) - (deltay(i,2) - deltay(i,1)) / 2
    
    for i in np.arange(1,5+1).reshape(-1):
        for j in np.arange(2,4+1).reshape(-1):
            dy[i,j] = (deltay(i,j) + deltay(i,j - 1)) / 2
    
    for i in np.arange(1,5+1).reshape(-1):
        dy[i,5] = deltay(i,4) + (deltay(i,4) - deltay(i,3)) / 2
    
    for i in np.arange(1,4+1).reshape(-1):
        for j in np.arange(1,4+1).reshape(-1):
            hni = (x(5) - x(1)) / (x(i + 1) - x(i))
            lmj = (y(5) - y(1)) / (y(j + 1) - y(j))
            deltn1x = (f(5,1) - f(1,1)) / (x(i + 1) - x(i))
            deltnmx = (f(5,5) - f(1,5)) / (x(i + 1) - x(i))
            delt1my = (f(1,5) - f(1,1)) / (y(j + 1) - y(j))
            deltnmy = (f(5,5) - f(5,1)) / (y(j + 1) - y(j))
    
    N = 5
    M = 5
    for i in np.arange(1,4+1).reshape(-1):
        for j in np.arange(1,4+1).reshape(-1):
            s1[i,j] = np.amin(dx(i,j) / (hni * dx(1,1)),dx(i + 1,j) / (hni * dx(N,1)))
            s2[i,j] = np.amin(dx(i,j + 1) / (hni * dx(1,M)),dx(i + 1,j + 1) / (hni * dx(N,M)))
            s3[i,j] = np.amin(deltax(i,j) / (deltn1x),deltax(i,j + 1) / (deltnmx))
            s4[i,j] = np.amin(dy(i,j) / (lmj * dy(1,1)),dy(i,j + 1) / (lmj * dy(1,M)))
            s5[i,j] = np.amin(dy(i + 1,j) / (lmj * dy(N,1)),dy(i + 1,j + 1) / (lmj * dy(N,M)))
            s6[i,j] = np.amin(deltay(i,j) / (delt1my),deltay(i + 1,j) / (deltnmy))
            ss1[i,j] = np.amin(s1(i,j),s2(i,j))
            ss2[i,j] = np.amin(s3(i,j),s4(i,j))
            ss3[i,j] = np.amin(s5(i,j),s6(i,j))
            sss1[i,j] = np.amin(ss1(i,j),ss2(i,j))
            sss2[i,j] = np.amin(0.2,ss3(i,j))
            s[i,j] = 0.5 * np.amin(sss1(i,j),sss2(i,j))
    
    for i in np.arange(1,4+1).reshape(-1):
        for j in np.arange(1,4+1).reshape(-1):
            xga1[i,j] = np.amax(2 * (dx(i,j) - s(i,j) * hni * dx(1,1)) / (deltax(i,j) - s(i,j) * deltn1x),2 * (dx(i,j + 1) - s(i,j) * hni * dx(1,M)) / (deltax(i,j + 1) - s(i,j) * deltnmx))
            xga[i,j] = 0.5 + np.amax(xga1(i,j),0)
            xga[i,5] = 0.5 + np.amax(2 * (dx(i,5) - s(i,4) * hni * dx(1,5)) / (deltax(i,5) - s(i,4) * deltnmx),0)
            xgb1[i,j] = np.amax(2 * (dx(i + 1,j) - s(i,j) * hni * dx(N,1)) / (deltax(i,j) - s(i,j) * deltn1x),2 * (dx(i + 1,j + 1) - s(i,j) * hni * dx(N,M)) / (deltax(i,j + 1) - s(i,j) * deltnmx))
            xgb[i,j] = 0.5 + np.amax(xgb1(i,j),0)
            xgb[i,5] = 0.5 + np.amax(2 * (dx(i + 1,5) - s(i,4) * hni * dx(N,5)) / (deltax(i,5) - s(i,4) * deltnmx),0)
            yga1[1,j] = np.amax(2 * (dy(1,j) - s(1,j) * lmj * dy(1,1)) / (deltay(1,j) - s(1,j) * delt1my),2 * (dy(2,j) - s(1,j) * lmj * dy(N,1)) / (deltay(2,j) - s(1,j) * deltnmy))
            yga2[2,j] = np.amax(2 * (dy(2,j) - s(2,j) * lmj * dy(1,1)) / (deltay(2,j) - s(2,j) * delt1my),2 * (dy(3,j) - s(2,j) * lmj * dy(N,1)) / (deltay(3,j) - s(2,j) * deltnmy))
            yga3[3,j] = np.amax(2 * (dy(3,j) - s(3,j) * lmj * dy(1,1)) / (deltay(3,j) - s(3,j) * delt1my),2 * (dy(4,j) - s(3,j) * lmj * dy(N,1)) / (deltay(4,j) - s(3,j) * deltnmy))
            yga4[4,j] = np.amax(2 * (dy(4,j) - s(4,j) * lmj * dy(1,1)) / (deltay(4,j) - s(4,j) * delt1my),2 * (dy(5,j) - s(4,j) * lmj * dy(N,1)) / (deltay(5,j) - s(4,j) * deltnmy))
            yyga1[i,j] = np.amax(yga1(1,j),yga2(2,j))
            yyga2[i,j] = np.amax(yga3(3,j),yga4(4,j))
            yga[i,j] = 0.0 + np.amax(yyga1(i,j),yyga2(i,j))
            ygb1[1,j] = np.amax(2 * (dy(1,j + 1) - s(1,j) * lmj * dy(1,M)) / (deltay(1,j) - s(1,j) * delt1my),2 * (dy(2,j + 1) - s(1,j) * lmj * dy(N,M)) / (deltay(2,j) - s(1,j) * deltnmy))
            ygb2[2,j] = np.amax(2 * (dy(2,j + 1) - s(2,j) * lmj * dy(1,M)) / (deltay(2,j) - s(2,j) * delt1my),2 * (dy(3,j + 1) - s(2,j) * lmj * dy(N,M)) / (deltay(3,j) - s(2,j) * deltnmy))
            ygb3[3,j] = np.amax(2 * (dy(3,j + 1) - s(3,j) * lmj * dy(1,M)) / (deltay(3,j) - s(3,j) * delt1my),2 * (dy(4,j + 1) - s(3,j) * lmj * dy(N,M)) / (deltay(4,j) - s(3,j) * deltnmy))
            ygb4[4,j] = np.amax(2 * (dy(4,j + 1) - s(4,j) * lmj * dy(1,M)) / (deltay(4,j) - s(4,j) * delt1my),2 * (dy(5,j + 1) - s(4,j) * lmj * dy(N,M)) / (deltay(5,j) - s(4,j) * deltnmy))
            yygb1[i,j] = np.amax(ygb1(1,j),ygb2(2,j))
            yygb2[i,j] = np.amax(ygb3(3,j),ygb4(4,j))
            ygb[i,j] = 0.0 + np.amax(yygb1(i,j),yygb2(i,j))
    