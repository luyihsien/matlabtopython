    
def dy(i = None,j = None,y = None,ppp = None,z = None): 
    z = z
    if (j == 1):
        res = (4 * z(i,2) - 3 * z(i,1) - z(i,3)) / (2 * (y(j + 1) - y(j)))
    else:
        if (j <=(ppp - 1)):
            res = ((z(i,j + 1) - z(i,j - 1))) / (2 * (y(j + 1) - y(j)))
        else:
            res = (3 * z(i,ppp) - 4 * z(i,ppp - 1) + z(i,ppp - 2)) / (2 * (y(j) - y(j - 1)))
    
    return res