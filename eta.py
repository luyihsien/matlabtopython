    
def eta(j = None,y = None,pp = None): 
    pp = 4
    res = (y(j) - y(1)) / (y(pp + 1) - y(1))
    return res