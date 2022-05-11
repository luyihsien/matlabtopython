import numpy as np
    
def downsample(input_args = None): 
    #LR can be obtained
    m,n,ch = input_args.shape
    if ch == 3:
        im_r = input_args(:,:,1)
        im_g = input_args(:,:,2)
        im_b = input_args(:,:,3)
        im_r_l = fdownsampling(im_r)
        im_g_l = fdownsampling(im_g)
        im_b_l = fdownsampling(im_b)
        nrow,ncol = im_r_l.shape
        output_args = np.zeros((np.array([nrow,ncol,3]),np.array([nrow,ncol,3])))
        output_args[:,:,1] = im_r_l
        output_args[:,:,2] = im_g_l
        output_args[:,:,3] = im_b_l
        output_args = uint8(output_args)
    else:
        output_args = fdownsampling(input_args)
        output_args = uint8(output_args)
    
    return output_args
    
    return output_args