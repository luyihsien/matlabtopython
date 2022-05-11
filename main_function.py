import numpy as np
import fractal_interpolation2
import blanket
def main_function(II = None,m = None,n = None,scale = None): 
    #MAIN_FUNCTION Summary of this function goes here
#   Detailed explanation goes here
    if scale == 2:
        for i in np.arange(1,m+2,2).reshape(-1):
            for j in np.arange(1,n+2,2).reshape(-1):
                block = II(np.arange(i,i + 4+1),np.arange(j,j + 4+1))
                block=block.astype(double)
                T = blanket(block)
                if (T > 2.12):
                    #rational fractal interpolation
                    c = fractal_interpolation2(block)
                else:
                    #rational interpolation
                    c = fractal_interpolation2_s0(block)
                if (j == 1):
                    A = c(np.arange(1,4+1),np.arange(1,4+1))
                else:
                    A = cat(2,A,c(np.arange(1,4+1),np.arange(1,4+1)))
            if (i == 1):
                B = A
            else:
                B = cat(1,B,A)
    
    if scale == 3:
        for i in np.arange(1,m+2,2).reshape(-1):
            for j in np.arange(1,n+2,2).reshape(-1):
                block = double(II(np.arange(i,i + 4+1),np.arange(j,j + 4+1)))
                T = blanket(block)
                if (T > 2.21):
                    #rational fractal interpolation
                    c = fractal_interpolation3(block)
                else:
                    #rational interpolation
                    c = fractal_interpolation3_s0(block)
                if (j == 1):
                    A = c(np.arange(1,6+1),np.arange(1,6+1))
                else:
                    A = cat(2,A,c(np.arange(1,6+1),np.arange(1,6+1)))
            if (i == 1):
                B = A
            else:
                B = cat(1,B,A)
        #    B=B(1:end-4,1:end-4);
    
    if scale == 4:
        for i in np.arange(1,m+2,2).reshape(-1):
            for j in np.arange(1,n+2,2).reshape(-1):
                block = double(II(np.arange(i,i + 4+1),np.arange(j,j + 4+1)))
                T = blanket(block)
                if (T > 2.26):
                    c = fractal_interpolation4(block)
                else:
                    c = fractal_interpolation4_s0(block)
                if (j == 1):
                    A = c(np.arange(1,8+1),np.arange(1,8+1))
                else:
                    A = cat(2,A,c(np.arange(1,8+1),np.arange(1,8+1)))
            if (i == 1):
                B = A
            else:
                B = cat(1,B,A)
    
    return B
    
    return B