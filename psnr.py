    
def psnr(f1 = None,f2 = None): 
    k = 8
    fmax = 2.0 ** k - 1
    a = fmax ** 2
    e = double(f1) - double(f2)
    m,n = e.shape
    b = sum(sum(e ** 2))
    PSNR = 10 * log10(m * n * a / b)
    return PSNR