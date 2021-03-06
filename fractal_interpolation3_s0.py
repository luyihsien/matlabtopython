import numpy as np
    
def fractal_interpolation3_s0(z = None): 
    s = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    alpha0 = np.array([[0.8,0.8,0.8,0.8],[0.8,0.8,0.8,0.8],[0.8,0.8,0.8,0.8],[0.8,0.8,0.8,0.8]])
    beta0 = np.array([[0.8,0.8,0.8,0.8],[0.8,0.8,0.8,0.8],[0.8,0.8,0.8,0.8],[0.8,0.8,0.8,0.8]])
    gama0 = np.array([[1.2,1.2,1.2,1.2],[1.2,1.2,1.2,1.2],[1.2,1.2,1.2,1.2],[1.2,1.2,1.2,1.2]])
    alpha = np.array([[0.2,0.2,0.6,0.6],[0.2,0.2,0.6,0.6],[0.6,0.6,0.6,0.6],[0.6,0.6,0.6,0.6]])
    beta = np.array([[0.2,0.2,0.6,0.6],[0.2,0.2,0.6,0.6],[0.6,0.6,0.6,0.6],[0.6,0.6,0.6,0.6]])
    gama = np.array([[2,2,1.5,1.5],[2,2,1.5,1.5],[1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5]])
    kk = 1
    M = 3
    N = 3
    x = np.arange(1,M+0.5,0.5)
    y = np.arange(1,N+0.5,0.5)
    m,n = z.shape
    ppp = m
    for p in np.arange(1,kk+1).reshape(-1):
        for i in np.arange(1,n - 1+1).reshape(-1):
            for k in np.arange(1,4+1).reshape(-1):
                for j in np.arange(1,m - 1+1).reshape(-1):
                    a = (x(i + 1) - x(i)) / (x(4 * int(np.floor((i - 1) / 4)) + 5) - x(4 * int(np.floor((i - 1) / 4)) + 1))
                    b = (x(4 * int(np.floor((i - 1) / 4)) + 5) * x(i) - x(4 * int(np.floor((i - 1) / 4)) + 1) * x(i + 1)) / (x(4 * int(np.floor((i - 1) / 4)) + 5) - x(4 * int(np.floor((i - 1) / 4)) + 1))
                    c = (y(j + 1) - y(j)) / (y(4 * int(np.floor((j - 1) / 4)) + 5) - y(4 * int(np.floor((j - 1) / 4)) + 1))
                    d = (y(4 * int(np.floor((j - 1) / 4)) + 5) * y(j) - y(4 * int(np.floor((j - 1) / 4)) + 1) * y(j + 1)) / (y(4 * int(np.floor((j - 1) / 4)) + 5) - y(4 * int(np.floor((j - 1) / 4)) + 1))
                    for t in np.arange(1,4+1).reshape(-1):
                        sita(k,x,n)
                        eta(t,y,m)
                        dx(i,j,x,ppp,z)
                        dy(i,j,y,ppp,z)
                        # sss=dx(k,t,x,ppp,z)-dy(k,t,y,ppp,z);
                        a00 = ((1 - sita(k,x,n)) ** 2 * (alpha0(i,j) + sita(k,x,n) * gama0(i,j)) * (1 - eta(t,y,m)) ** 2 * (alpha(i,j) + eta(t,y,m) * gama(i,j))) / (((1 - sita(k,x,n)) ** 2 * alpha0(i,j) + sita(k,x,n) * (1 - sita(k,x,n)) * gama0(i,j) + sita(k,x,n) ** 2 * beta0(i,j)) * ((1 - eta(t,y,m)) ** 2 * alpha(i,j) + eta(t,y,m) * (1 - eta(t,y,m)) * gama(i,j) + eta(t,y,m) ** 2 * beta(i,j)))
                        a01 = ((1 - sita(k,x,n)) ** 2 * (alpha0(i,j) + sita(k,x,n) * gama0(i,j)) * eta(t,y,m) ** 2 * (beta(i,j) + (1 - eta(t,y,m)) * gama(i,j))) / (((1 - sita(k,x,n)) ** 2 * alpha0(i,j) + sita(k,x,n) * (1 - sita(k,x,n)) * gama0(i,j) + sita(k,x,n) ** 2 * beta0(i,j)) * ((1 - eta(t,y,m)) ** 2 * alpha(i,j) + eta(t,y,m) * (1 - eta(t,y,m)) * gama(i,j) + eta(t,y,m) ** 2 * beta(i,j)))
                        a10 = (sita(k,x,n) ** 2 * (beta0(i,j) + (1 - sita(k,x,n)) * gama0(i,j)) * (1 - eta(t,y,m)) ** 2 * (alpha(i,j) + eta(t,y,m) * gama(i,j))) / (((1 - sita(k,x,n)) ** 2 * alpha0(i,j) + sita(k,x,n) * (1 - sita(k,x,n)) * gama0(i,j) + sita(k,x,n) ** 2 * beta0(i,j)) * ((1 - eta(t,y,m)) ** 2 * alpha(i,j) + eta(t,y,m) * (1 - eta(t,y,m)) * gama(i,j) + eta(t,y,m) ** 2 * beta(i,j)))
                        a11 = (sita(k,x,n) ** 2 * (beta0(i,j) + (1 - sita(k,x,n)) * gama0(i,j)) * (eta(t,y,m)) ** 2 * (beta(i,j) + (1 - eta(t,y,m)) * gama(i,j))) / (((1 - sita(k,x,n)) ** 2 * alpha0(i,j) + sita(k,x,n) * (1 - sita(k,x,n)) * gama0(i,j) + sita(k,x,n) ** 2 * beta0(i,j)) * ((1 - eta(t,y,m)) ** 2 * alpha(i,j) + eta(t,y,m) * (1 - eta(t,y,m)) * gama(i,j) + eta(t,y,m) ** 2 * beta(i,j)))
                        b00 = (sita(k,x,n) * (1 - sita(k,x,n)) ** 2 * alpha0(i,j) * (1 - eta(t,y,m)) ** 2 * (alpha(i,j) + eta(t,y,m) * gama(i,j))) / (((1 - sita(k,x,n)) ** 2 * alpha0(i,j) + sita(k,x,n) * (1 - sita(k,x,n)) * gama0(i,j) + sita(k,x,n) ** 2 * beta0(i,j)) * ((1 - eta(t,y,m)) ** 2 * alpha(i,j) + eta(t,y,m) * (1 - eta(t,y,m)) * gama(i,j) + eta(t,y,m) ** 2 * beta(i,j)))
                        b01 = (sita(k,x,n) * (1 - sita(k,x,n)) ** 2 * alpha0(i,j) * eta(t,y,m) ** 2 * (beta(i,j) + (1 - eta(t,y,m)) * gama(i,j))) / (((1 - sita(k,x,n)) ** 2 * alpha0(i,j) + sita(k,x,n) * (1 - sita(k,x,n)) * gama0(i,j) + sita(k,x,n) ** 2 * beta0(i,j)) * ((1 - eta(t,y,m)) ** 2 * alpha(i,j) + eta(t,y,m) * (1 - eta(t,y,m)) * gama(i,j) + eta(t,y,m) ** 2 * beta(i,j)))
                        b10 = - (sita(k,x,n) ** 2 * (1 - sita(k,x,n)) * beta0(i,j) * (1 - eta(t,y,m)) ** 2 * (alpha(i,j) + eta(t,y,m) * gama(i,j))) / (((1 - sita(k,x,n)) ** 2 * alpha0(i,j) + sita(k,x,n) * (1 - sita(k,x,n)) * gama0(i,j) + sita(k,x,n) ** 2 * beta0(i,j)) * ((1 - eta(t,y,m)) ** 2 * alpha(i,j) + eta(t,y,m) * (1 - eta(t,y,m)) * gama(i,j) + eta(t,y,m) ** 2 * beta(i,j)))
                        b11 = - ((sita(k,x,n)) ** 2 * (1 - sita(k,x,n)) * beta0(i,j) * eta(t,y,m) ** 2 * (beta(i,j) + (1 - eta(t,y,m)) * gama(i,j))) / (((1 - sita(k,x,n)) ** 2 * alpha0(i,j) + sita(k,x,n) * (1 - sita(k,x,n)) * gama0(i,j) + sita(k,x,n) ** 2 * beta0(i,j)) * ((1 - eta(t,y,m)) ** 2 * alpha(i,j) + eta(t,y,m) * (1 - eta(t,y,m)) * gama(i,j) + eta(t,y,m) ** 2 * beta(i,j)))
                        c00 = ((1 - sita(k,x,n)) ** 2 * (alpha0(i,j) + sita(k,x,n) * gama0(i,j)) * eta(t,y,m) * (1 - eta(t,y,m)) ** 2 * alpha(i,j)) / (((1 - sita(k,x,n)) ** 2 * alpha0(i,j) + sita(k,x,n) * (1 - sita(k,x,n)) * gama0(i,j) + sita(k,x,n) ** 2 * beta0(i,j)) * ((1 - eta(t,y,m)) ** 2 * alpha(i,j) + eta(t,y,m) * (1 - eta(t,y,m)) * gama(i,j) + eta(t,y,m) ** 2 * beta(i,j)))
                        c01 = - ((1 - sita(k,x,n)) ** 2 * (alpha0(i,j) + sita(k,x,n) * gama0(i,j)) * eta(t,y,m) ** 2 * (1 - eta(t,y,m)) * beta(i,j)) / (((1 - sita(k,x,n)) ** 2 * alpha0(i,j) + sita(k,x,n) * (1 - sita(k,x,n)) * gama0(i,j) + sita(k,x,n) ** 2 * beta0(i,j)) * ((1 - eta(t,y,m)) ** 2 * alpha(i,j) + eta(t,y,m) * (1 - eta(t,y,m)) * gama(i,j) + eta(t,y,m) ** 2 * beta(i,j)))
                        c10 = (sita(k,x,n) ** 2 * (beta0(i,j) + (1 - sita(k,x,n)) * gama0(i,j)) * eta(t,y,m) * (1 - eta(t,y,m)) ** 2 * alpha(i,j)) / (((1 - sita(k,x,n)) ** 2 * alpha0(i,j) + sita(k,x,n) * (1 - sita(k,x,n)) * gama0(i,j) + sita(k,x,n) ** 2 * beta0(i,j)) * ((1 - eta(t,y,m)) ** 2 * alpha(i,j) + eta(t,y,m) * (1 - eta(t,y,m)) * gama(i,j) + eta(t,y,m) ** 2 * beta(i,j)))
                        c11 = - (sita(k,x,n) ** 2 * (beta0(i,j) + (1 - sita(k,x,n)) * gama0(i,j)) * eta(t,y,m) ** 2 * (1 - eta(t,y,m)) * beta(i,j)) / (((1 - sita(k,x,n)) ** 2 * alpha0(i,j) + sita(k,x,n) * (1 - sita(k,x,n)) * gama0(i,j) + sita(k,x,n) ** 2 * beta0(i,j)) * ((1 - eta(t,y,m)) ** 2 * alpha(i,j) + eta(t,y,m) * (1 - eta(t,y,m)) * gama(i,j) + eta(t,y,m) ** 2 * beta(i,j)))
                        xx[[i - 1] * 4 + t] = a * x(4 * int(np.floor((i - 1) / 4)) + t) + b
                        yy[[j - 1] * 4 + t] = c * y(4 * int(np.floor((j - 1) / 4)) + t) + d
                        aa[[i - 1] * 4 + k,[j - 1] * 4 + t] = alpha(k,t)
                        bb[[i - 1] * 4 + k,[j - 1] * 4 + t] = beta(k,t)
                        ss[[i - 1] * 4 + k,[j - 1] * 4 + t] = s(k,t)
                        zz[[i - 1] * 4 + k,[j - 1] * 4 + t] = s(i,j) * z(k,t) + a00 * (z(i,j) - s(i,j) * z(1,1)) + a10 * (z(i + 1,j) - s(i,j) * z(ppp,1)) + a01 * (z(i,j + 1) - s(i,j) * z(1,ppp)) + a11 * (z(i + 1,j + 1) - s(i,j) * z(ppp,ppp)) + b00 * ((x(i + 1) - x(i)) * dx(i,j,x,ppp,z) - s(i,j) * (x(ppp) - x(1)) * dx(1,1,x,ppp,z)) + b10 * ((x(i + 1) - x(i)) * dx(i + 1,j,x,ppp,z) - s(i,j) * (x(ppp) - x(1)) * dx(ppp,1,x,ppp,z)) + b01 * ((x(i + 1) - x(i)) * dx(i,j + 1,x,ppp,z) - s(i,j) * (x(ppp) - x(1)) * dx(1,ppp,x,ppp,z)) + b11 * ((x(i + 1) - x(i)) * dx(i + 1,j + 1,x,ppp,z) - s(i,j) * (x(ppp) - x(1)) * dx(ppp,ppp,x,ppp,z)) + c00 * ((y(j + 1) - y(j)) * dy(i,j,y,ppp,z) - s(i,j) * (y(ppp) - y(1)) * dy(1,1,y,ppp,z)) + c10 * ((y(j + 1) - y(j)) * dy(i + 1,j,y,ppp,z) - s(i,j) * (y(ppp) - y(1)) * dy(ppp,1,y,ppp,z)) + c01 * ((y(j + 1) - y(j)) * dy(i,j + 1,y,ppp,z) - s(i,j) * (y(ppp) - y(1)) * dy(1,ppp,y,ppp,z)) + c11 * ((y(j + 1) - y(j)) * dy(i + 1,j + 1,y,ppp,z) - s(i,j) * (y(ppp) - y(1)) * dy(ppp,ppp,y,ppp,z))
        zx = z(n,np.arange(1,m+1))
        zy = z(np.arange(1,n+1),m)
        for i in np.arange(1,n - 1+1).reshape(-1):
            for j in np.arange(1,4+1).reshape(-1):
                sita(j,x,n)
                eta(j,y,m)
                aa00 = ((1 - sita(j,x,n)) ** 2 * (alpha0(i,m - 1) + sita(j,x,n) * gama0(i,m - 1))) / ((1 - sita(j,x,n)) ** 2 * alpha0(i,m - 1) + sita(j,x,n) * (1 - sita(j,x,n)) * gama0(i,m - 1) + sita(j,x,n) ** 2 * beta0(i,m - 1))
                aa10 = (sita(j,x,n) ** 2 * ((1 - sita(j,x,n)) * gama0(i,m - 1) + beta0(i,m - 1))) / ((1 - sita(j,x,n)) ** 2 * alpha0(i,m - 1) + sita(j,x,n) * (1 - sita(j,x,n)) * gama0(i,m - 1) + sita(j,x,n) ** 2 * beta0(i,m - 1))
                bb10 = (sita(j,x,n) * (1 - sita(j,x,n)) ** 2 * alpha0(i,m - 1)) / ((1 - sita(j,x,n)) ** 2 * alpha0(i,m - 1) + sita(j,x,n) * (1 - sita(j,x,n)) * gama0(i,m - 1) + sita(j,x,n) ** 2 * beta0(i,m - 1))
                bb11 = - ((sita(j,x,n)) ** 2 * (1 - sita(j,x,n)) * beta(i,m - 1)) / ((1 - sita(j,x,n)) ** 2 * alpha0(i,m - 1) + sita(j,x,n) * (1 - sita(j,x,n)) * gama0(i,m - 1) + sita(j,x,n) ** 2 * beta0(i,m - 1))
                zzx[[i - 1] * 4 + j] = s(i,m - 1) * zx(4 * int(np.floor((i - 1) / 4)) + j) + aa00 * (zx(i) - s(i,m - 1) * zx(4 * int(np.floor((i - 1) / 4)) + 1)) + aa10 * (zx(i + 1) - s(i,m - 1) * zx(4 * int(np.floor((i - 1) / 4)) + 5)) + bb10 * ((x(i + 1) - x(i)) * dx(i,m,x,m,z) - s(i,m - 1) * (x(4 * int(np.floor((i - 1) / 4)) + 5) - x(4 * int(np.floor((i - 1) / 4)) + 1)) * dx(1,m,x,m,z)) + bb11 * ((x(i + 1) - x(i)) * dx(i + 1,m,x,n,z) - s(i,m - 1) * (x(4 * int(np.floor((i - 1) / 4)) + 5) - x(4 * int(np.floor((i - 1) / 4)) + 1)) * dx(n,m,x,n,z))
                a00 = ((1 - eta(j,y,m)) ** 2 * (alpha(n - 1,i) + eta(j,y,m) * gama(n - 1,i))) / ((1 - eta(j,y,m)) ** 2 * alpha(n - 1,i) + eta(j,y,m) * (1 - eta(j,y,m)) * gama(n - 1,i) + eta(j,y,m) ** 2 * beta(n - 1,i))
                a01 = (eta(j,y,m) ** 2 * ((1 - eta(j,y,m)) * gama(n - 1,i) + beta(n - 1,i))) / ((1 - eta(j,y,m)) ** 2 * alpha(n - 1,i) + eta(j,y,m) * (1 - eta(j,y,m)) * gama(n - 1,i) + eta(j,y,m) ** 2 * beta(n - 1,i))
                c10 = (eta(j,y,m) * (1 - eta(j,y,m)) ** 2 * alpha(n - 1,i)) / ((1 - eta(j,y,m)) ** 2 * alpha(n - 1,i) + eta(j,y,m) * (1 - eta(j,y,m)) * gama(n - 1,i) + eta(j,y,m) ** 2 * beta(n - 1,i))
                c11 = - ((1 - eta(j,y,m)) * eta(j,y,m) ** 2 * beta(n - 1,i)) / ((1 - eta(j,y,m)) ** 2 * alpha(n - 1,i) + eta(j,y,m) * (1 - eta(j,y,m)) * gama(n - 1,i) + eta(j,y,m) ** 2 * beta(n - 1,i))
                zzy[[i - 1] * 4 + j] = s(n - 1,i) * zy(4 * int(np.floor((i - 1) / 4)) + j) + a00 * (zy(i) - s(n - 1,i) * zy(4 * int(np.floor((i - 1) / 4)) + 1)) + a01 * (zy(i + 1) - s(n - 1,i) * zy(4 * int(np.floor((i - 1) / 4)) + 5)) + c10 * ((y(i + 1) - y(i)) * dy(n,i,y,m,z) - s(n - 1,i) * (y(4 * int(np.floor((i - 1) / 4)) + 5) - y(4 * int(np.floor((i - 1) / 4)) + 1)) * dy(n,1,y,m,z)) + c11 * ((y(i + 1) - y(i)) * dy(n,i + 1,y,n,z) - s(n - 1,i) * (y(4 * int(np.floor((i - 1) / 4)) + 5) - y(4 * int(np.floor((i - 1) / 4)) + 1)) * dy(n,m,y,n,z))
        zz = cat(2,zz,np.transpose(zzy))
        tmp = cat(2,zzx,z(n,m))
        zz = cat(1,zz,tmp)
        xx = cat(2,xx,xx(4 * (m - 1)) + xx(2) - xx(1))
        yy = cat(2,yy,yy(4 * (m - 1)) + yy(2) - yy(1))
        #=========================================
        x = xx
        y = yy
        z = zz
        alpha = aa
        beta = bb
        s = ss
        m,n = z.shape
    
    #pixel mapping
    
    ###############################################################################################################################
    average_final = np.zeros((12,12))
    # temp1=zeros(1,3);
# temp2=zeros(1,3);
# temp3=zeros(1,3);
# temp4=zeros(1,3);
    temp5 = np.zeros((1,4))
    temp6 = np.zeros((1,4))
    temp7 = np.zeros((1,4))
    temp8 = np.zeros((1,4))
    row = 1
    col = 1
    for i in np.arange(1,16+4,4).reshape(-1):
        for j in np.arange(1,16+4,4).reshape(-1):
            average_final[row,col] = zz(i,j)
            temp5[1,1] = zz(i,j)
            temp5[1,2] = zz(i,j + 1)
            temp5[1,3] = zz(i,j + 2)
            temp5[1,4] = zz(i,j + 3)
            max5 = np.amax(temp5)
            min5 = np.amin(temp5)
            ave1 = (max5 + min5) / 2
            average_final[row,col + 1] = ave1
            temp6[1,1] = zz(i,j + 1)
            temp6[1,2] = zz(i,j + 2)
            temp6[1,3] = zz(i,j + 3)
            temp6[1,4] = zz(i,j + 4)
            max6 = np.amax(temp6)
            min6 = np.amin(temp6)
            ave2 = (max6 + min6) / 2
            average_final[row,col + 2] = ave2
            temp7[1,1] = zz(i,j)
            temp7[1,2] = zz(i + 1,j)
            temp7[1,3] = zz(i + 2,j)
            temp7[1,4] = zz(i + 3,j)
            max7 = np.amax(temp7)
            min7 = np.amin(temp7)
            ave3 = (max7 + min7) / 2
            average_final[row + 1,col] = ave3
            temp8[1,1] = zz(i + 1,j)
            temp8[1,2] = zz(i + 2,j)
            temp8[1,3] = zz(i + 3,j)
            temp8[1,4] = zz(i + 4,j)
            max8 = np.amax(temp8)
            min8 = np.amin(temp8)
            ave4 = (max8 + min8) / 2
            average_final[row + 2,col] = ave4
            average_final[row + 1,col + 1] = (zz(i + 1,j + 1) + zz(i + 1,j + 2) + zz(i + 2,j + 1) + zz(i + 2,j + 2)) / 4
            average_final[row + 1,col + 2] = (zz(i + 1,j + 2) + zz(i + 1,j + 3) + zz(i + 2,j + 2) + zz(i + 2,j + 3)) / 4
            average_final[row + 2,col + 1] = (zz(i + 2,j + 1) + zz(i + 2,j + 2) + zz(i + 3,j + 1) + zz(i + 3,j + 2)) / 4
            average_final[row + 2,col + 2] = (zz(i + 2,j + 2) + zz(i + 2,j + 3) + zz(i + 3,j + 2) + zz(i + 3,j + 3)) / 4
            col = col + 3
            if (col == 13):
                row = row + 3
                col = 1
    
    return average_final
    
    return average_final