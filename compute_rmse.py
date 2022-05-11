import numpy as np
    
def compute_rmse(im1 = None,im2 = None): 
    if im1.shape[3-1] == 3:
        im1 = rgb2ycbcr(im1)
        im1 = im1(:,:,1)
    
    if im2.shape[3-1] == 3:
        im2 = rgb2ycbcr(im2)
        im2 = im2(:,:,1)
    
    imdff = double(im1) - double(im2)
    imdff = imdff
    rmse = np.sqrt(mean(imdff ** 2))
    return rmse