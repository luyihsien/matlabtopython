##
# Author: Yunfeng Zhang, Qinglan Fan, Fangxun Bao, etal.
# Date: 25/07/2018
# Title:Single-Image Super-Resolution Based on Rational Fractal Interpolation
##

#read image

import numpy as np
out_dir = '../results'
im_dir = '../data/'
im_name = 'bird_GT.bmp'
im = imread(fullfile(im_dir,im_name))
m,n,ch = im.shape
##  downsample
scale = 2

#select down-sampling method
dmethod = 1
if dmethod == 1:
    # bicubic down-sampling
    LR = imresize(im,1 / scale,'bicubic')

if dmethod == 2:
    # average down-samping
    LR = downsample(im)

if dmethod == 3:
    # LR images are obtained by down-sampling the HR images directly along both the horizontal and vertical directions by a factor of 2, 3, or 4.
    LR = im(np.arange(1,end()+scale,scale),np.arange(1,end()+scale,scale),:)

##
if ch == 3:
    # change color space, work on illuminance only
    im_l_ycbcr = rgb2ycbcr(LR)
    im_l_y = im_l_ycbcr(:,:,1)
    im_l_cb = im_l_ycbcr(:,:,2)
    im_l_cr = im_l_ycbcr(:,:,3)
    im_l_y = double(im_l_y)
    im_l_cb = double(im_l_cb)
    im_l_cr = double(im_l_cr)
    #expand the metrix
    m,n = im_l_y.shape
    II[np.arange[1,m+1],np.arange[1,n+1]] = im_l_y
    II[m + 1,:] = 2.0 * II(m,:) - II(m - 1,:)
    II[:,n + 1] = 2.0 * II(:,n) - II(:,n - 1)
    II[m + 2,:] = 2.0 * II(m + 1,:) - II(m,:)
    II[:,n + 2] = 2.0 * II(:,n + 1) - II(:,n)
    II[m + 3,:] = 2.0 * II(m + 2,:) - II(m + 1,:)
    II[:,n + 3] = 2.0 * II(:,n + 2) - II(:,n + 1)
    II[m + 4,:] = 2.0 * II(m + 3,:) - II(m + 2,:)
    II[:,n + 4] = 2.0 * II(:,n + 3) - II(:,n + 2)
    # image super-resolution
    im_h_y = main_function(II,m,n,scale)
    # upscale the chrominance simply by "bicubic"
    nrow,ncol = im_h_y.shape
    im_h_cb = imresize(im_l_cb,np.array([nrow,ncol]),'bicubic')
    im_h_cr = imresize(im_l_cr,np.array([nrow,ncol]),'bicubic')
    im_h_ycbcr = np.zeros((np.array([nrow,ncol,3]),np.array([nrow,ncol,3])))
    im_h_ycbcr[:,:,1] = im_h_y
    im_h_ycbcr[:,:,2] = im_h_cb
    im_h_ycbcr[:,:,3] = im_h_cr
    im_h = ycbcr2rgb(uint8(im_h_ycbcr))
    if dmethod == 1 or dmethod == 2:
        Image = double(im_h)
        Image1 = Image
        Half_size = 2
        F_size = 2 * Half_size + 1
        G_Filter = fspecial('gaussian',F_size,F_size / 6)
        Image_Filter = imfilter(Image1,G_Filter,'conv')
        Image_Diff = Image - Image_Filter
        Image_out = Image_Diff + 0
        Image1 = Image + Image_out
        im_h = uint8(Image1)
    # show the images
    figure
    imshow(im_h)
    fname = strcat('SRRFL_',im_name)
    imwrite(im_h,fullfile(out_dir,fname))
else:
    I = double(LR)
    #expand the metrix
    m,n = I.shape
    II[np.arange[1,m+1],np.arange[1,n+1]] = I
    II[m + 1,:] = 2.0 * II(m,:) - II(m - 1,:)
    II[:,n + 1] = 2.0 * II(:,n) - II(:,n - 1)
    II[m + 2,:] = 2.0 * II(m + 1,:) - II(m,:)
    II[:,n + 2] = 2.0 * II(:,n + 1) - II(:,n)
    II[m + 3,:] = 2.0 * II(m + 2,:) - II(m + 1,:)
    II[:,n + 3] = 2.0 * II(:,n + 2) - II(:,n + 1)
    II[m + 4,:] = 2.0 * II(m + 3,:) - II(m + 2,:)
    II[:,n + 4] = 2.0 * II(:,n + 3) - II(:,n + 2)
    # image super-resolution
    im_h = main_function(II,m,n,scale)
    im_h = uint8(im_h)
    if dmethod == 1 or dmethod == 2:
        Image = double(im_h)
        Image1 = Image
        Half_size = 2
        F_size = 2 * Half_size + 1
        G_Filter = fspecial('gaussian',F_size,F_size / 6)
        Image_Filter = imfilter(Image1,G_Filter,'conv')
        Image_Diff = Image - Image_Filter
        Image_out = Image_Diff + 0
        Image1 = Image + Image_out
        im_h = uint8(Image1)
    # show the images
    figure
    imshow(im_h)
    fname = strcat('SRRFL_',im_name)
    imwrite(im_h,fullfile(out_dir,fname))

##