import numpy as np
    
def FeatureSIM(imageRef = None,imageDis = None): 
    # ========================================================================
# FSIM Index with automatic downsampling, Version 1.0
# Copyright(c) 2010 Lin ZHANG, Lei Zhang, Xuanqin Mou and David Zhang
# All Rights Reserved.
    
    # ----------------------------------------------------------------------
# Permission to use, copy, or modify this software and its documentation
# for educational and research purposes only and without fee is here
# granted, provided that this copyright notice and the original authors'
# names appear on all copies and supporting documentation. This program
# shall not be used, rewritten, or adapted as the basis of a commercial
# software or hardware product without first obtaining permission of the
# authors. The authors make no representations about the suitability of
# this software for any purpose. It is provided "as is" without express
# or implied warranty.
#----------------------------------------------------------------------
    
    # This is an implementation of the algorithm for calculating the
# Feature SIMilarity (FSIM) index between two images.
    
    # Please refer to the following paper
    
    # Lin Zhang, Lei Zhang, Xuanqin Mou, and David Zhang,"FSIM: a feature similarity
# index for image qualtiy assessment", IEEE Transactions on Image Processing, vol. 20, no. 8, pp. 2378-2386, 2011.
    
    #----------------------------------------------------------------------
    
    #Input : (1) imageRef: the first image being compared
#        (2) imageDis: the second image being compared
    
    #Output: (1) FSIM: is the similarty score calculated using FSIM algorithm. FSIM
#	     only considers the luminance component of images. For colorful images,
#            they will be converted to the grayscale at first.
#        (2) FSIMc: is the similarity score calculated using FSIMc algorithm. FSIMc
#            considers both the grayscale and the color information.
#Note: For grayscale images, the returned FSIM and FSIMc are the same.
    
    #-----------------------------------------------------------------------
    
    #Usage:
#Given 2 test images img1 and img2. For gray-scale images, their dynamic range should be 0-255.
#For colorful images, the dynamic range of each color channel should be 0-255.
    
    #[FSIM, FSIMc] = FeatureSIM(img1, img2);
#-----------------------------------------------------------------------
    
    rows,cols = imageRef(:,:,1).shape
    I1 = np.ones((rows,cols))
    I2 = np.ones((rows,cols))
    Q1 = np.ones((rows,cols))
    Q2 = np.ones((rows,cols))
    if np.asarray(imageRef).ndim == 3:
        Y1 = 0.299 * double(imageRef(:,:,1)) + 0.587 * double(imageRef(:,:,2)) + 0.114 * double(imageRef(:,:,3))
        Y2 = 0.299 * double(imageDis(:,:,1)) + 0.587 * double(imageDis(:,:,2)) + 0.114 * double(imageDis(:,:,3))
        I1 = 0.596 * double(imageRef(:,:,1)) - 0.274 * double(imageRef(:,:,2)) - 0.322 * double(imageRef(:,:,3))
        I2 = 0.596 * double(imageDis(:,:,1)) - 0.274 * double(imageDis(:,:,2)) - 0.322 * double(imageDis(:,:,3))
        Q1 = 0.211 * double(imageRef(:,:,1)) - 0.523 * double(imageRef(:,:,2)) + 0.312 * double(imageRef(:,:,3))
        Q2 = 0.211 * double(imageDis(:,:,1)) - 0.523 * double(imageDis(:,:,2)) + 0.312 * double(imageDis(:,:,3))
    else:
        Y1 = imageRef
        Y2 = imageDis
    
    Y1 = double(Y1)
    Y2 = double(Y2)
    #########################
# Downsample the image
#########################
    minDimension = np.amin(rows,cols)
    F = np.amax(1,np.round(minDimension / 256))
    aveKernel = fspecial('average',F)
    aveI1 = conv2(I1,aveKernel,'same')
    aveI2 = conv2(I2,aveKernel,'same')
    I1 = aveI1(np.arange(1,rows+F,F),np.arange(1,cols+F,F))
    I2 = aveI2(np.arange(1,rows+F,F),np.arange(1,cols+F,F))
    aveQ1 = conv2(Q1,aveKernel,'same')
    aveQ2 = conv2(Q2,aveKernel,'same')
    Q1 = aveQ1(np.arange(1,rows+F,F),np.arange(1,cols+F,F))
    Q2 = aveQ2(np.arange(1,rows+F,F),np.arange(1,cols+F,F))
    aveY1 = conv2(Y1,aveKernel,'same')
    aveY2 = conv2(Y2,aveKernel,'same')
    Y1 = aveY1(np.arange(1,rows+F,F),np.arange(1,cols+F,F))
    Y2 = aveY2(np.arange(1,rows+F,F),np.arange(1,cols+F,F))
    #########################
# Calculate the phase congruency maps
#########################
    PC1 = phasecong2(Y1)
    PC2 = phasecong2(Y2)
    #########################
# Calculate the gradient map
#########################
    dx = np.array([[3,0,- 3],[10,0,- 10],[3,0,- 3]]) / 16
    dy = np.array([[3,10,3],[0,0,0],[- 3,- 10,- 3]]) / 16
    IxY1 = conv2(Y1,dx,'same')
    IyY1 = conv2(Y1,dy,'same')
    gradientMap1 = np.sqrt(IxY1 ** 2 + IyY1 ** 2)
    IxY2 = conv2(Y2,dx,'same')
    IyY2 = conv2(Y2,dy,'same')
    gradientMap2 = np.sqrt(IxY2 ** 2 + IyY2 ** 2)
    #########################
# Calculate the FSIM
#########################
    T1 = 0.85
    
    T2 = 160
    
    PCSimMatrix = (np.multiply(2 * PC1,PC2) + T1) / (PC1 ** 2 + PC2 ** 2 + T1)
    gradientSimMatrix = (np.multiply(2 * gradientMap1,gradientMap2) + T2) / (gradientMap1 ** 2 + gradientMap2 ** 2 + T2)
    PCm = np.amax(PC1,PC2)
    SimMatrix = np.multiply(np.multiply(gradientSimMatrix,PCSimMatrix),PCm)
    FSIM = sum(sum(SimMatrix)) / sum(sum(PCm))
    #########################
# Calculate the FSIMc
#########################
    T3 = 200
    T4 = 200
    ISimMatrix = (np.multiply(2 * I1,I2) + T3) / (I1 ** 2 + I2 ** 2 + T3)
    QSimMatrix = (np.multiply(2 * Q1,Q2) + T4) / (Q1 ** 2 + Q2 ** 2 + T4)
    lambda_ = 0.03
    SimMatrixC = np.multiply(np.multiply(np.multiply(gradientSimMatrix,PCSimMatrix),real((np.multiply(ISimMatrix,QSimMatrix)) ** lambda_)),PCm)
    FSIMc = sum(sum(SimMatrixC)) / sum(sum(PCm))
    return FSIM,FSIMc
    ##################################################################################
    
    
def phasecong2(im = None): 
    # ========================================================================
# Copyright (c) 1996-2009 Peter Kovesi
# School of Computer Science & Software Engineering
# The University of Western Australia
# http://www.csse.uwa.edu.au/
    
    # Permission is hereby  granted, free of charge, to any  person obtaining a copy
# of this software and associated  documentation files (the "Software"), to deal
# in the Software without restriction, subject to the following conditions:
    
    # The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
    
    # The software is provided "as is", without warranty of any kind.
# References:
    
    #     Peter Kovesi, "Image Features From Phase Congruency". Videre: A
#     Journal of Computer Vision Research. MIT Press. Volume 1, Number 3,
#     Summer 1999 http://mitpress.mit.edu/e-journals/Videre/001/v13.html
    
    nscale = 4
    
    norient = 4
    
    minWaveLength = 6
    
    mult = 2
    
    sigmaOnf = 0.55
    
    # Gaussian describing the log Gabor filter's
# transfer function in the frequency domain
# to the filter center frequency.
    dThetaOnSigma = 1.2
    
    # and the standard deviation of the angular Gaussian
# function used to construct filters in the
# freq. plane.
    k = 2.0
    
    # energy beyond the mean at which we set the
# noise threshold point.
# below which phase congruency values get
# penalized.
    epsilon = 0.0001
    
    thetaSigma = pi / norient / dThetaOnSigma
    
    # angular Gaussian function used to
# construct filters in the freq. plane.
    
    rows,cols = im.shape
    imagefft = fft2(im)
    
    zero = np.zeros((rows,cols))
    EO = cell(nscale,norient)
    
    estMeanE2n = []
    ifftFilterArray = cell(1,nscale)
    
    # Pre-compute some stuff to speed up filter construction
    
    # Set up X and Y matrices with ranges normalised to +/- 0.5
# The following code adjusts things appropriately for odd and even values
# of rows and columns.
    if np.mod(cols,2):
        xrange = np.array([np.arange(- (cols - 1) / 2,(cols - 1) / 2+1)]) / (cols - 1)
    else:
        xrange = np.array([np.arange(- cols / 2,(cols / 2 - 1)+1)]) / cols
    
    if np.mod(rows,2):
        yrange = np.array([np.arange(- (rows - 1) / 2,(rows - 1) / 2+1)]) / (rows - 1)
    else:
        yrange = np.array([np.arange(- rows / 2,(rows / 2 - 1)+1)]) / rows
    
    x,y = np.meshgrid(xrange,yrange)
    radius = np.sqrt(x ** 2 + y ** 2)
    
    theta = atan2(- y,x)
    
    # (note -ve y is used to give +ve
# anti-clockwise angles)
    
    radius = ifftshift(radius)
    
    theta = ifftshift(theta)
    
    radius[1,1] = 1
    
    # frequency point (now at top-left corner)
# so that taking the log of the radius will
# not cause trouble.
    
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    clear('x')
    clear('y')
    clear('theta')
    
    # Filters are constructed in terms of two components.
# 1) The radial component, which controls the frequency band that the filter
#    responds to
# 2) The angular component, which controls the orientation that the filter
#    responds to.
# The two components are multiplied together to construct the overall filter.
    
    # Construct the radial filter components...
    
    # First construct a low-pass filter that is as large as possible, yet falls
# away to zero at the boundaries.  All log Gabor filters are multiplied by
# this to ensure no extra frequencies at the 'corners' of the FFT are
# incorporated as this seems to upset the normalisation process when
# calculating phase congrunecy.
    lp = lowpassfilter(np.array([rows,cols]),0.45,15)
    
    logGabor = cell(1,nscale)
    for s in np.arange(1,nscale+1).reshape(-1):
        wavelength = minWaveLength * mult ** (s - 1)
        fo = 1.0 / wavelength
        logGabor[s] = np.exp((- (np.log(radius / fo)) ** 2) / (2 * np.log(sigmaOnf) ** 2))
        logGabor[s] = np.multiply(logGabor[s],lp)
        logGabor[s][1,1] = 0
        # back to zero (undo the radius fudge).
    
    # Then construct the angular filter components...
    
    spread = cell(1,norient)
    for o in np.arange(1,norient+1).reshape(-1):
        angl = (o - 1) * pi / norient
        # For each point in the filter matrix calculate the angular distance from
# the specified filter orientation.  To overcome the angular wrap-around
# problem sine difference and cosine difference values are first computed
# and then the atan2 function is used to determine angular distance.
        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
        dtheta = np.abs(atan2(ds,dc))
        spread[o] = np.exp((- dtheta ** 2) / (2 * thetaSigma ** 2))
        # angular filter component.
    
    # The main loop...
    EnergyAll[rows,cols] = 0
    AnAll[rows,cols] = 0
    for o in np.arange(1,norient+1).reshape(-1):
        sumE_ThisOrient = zero
        sumO_ThisOrient = zero
        sumAn_ThisOrient = zero
        Energy = zero
        for s in np.arange(1,nscale+1).reshape(-1):
            filter = np.multiply(logGabor[s],spread[o])
            # components to get the filter.
            ifftFilt = real(ifft2(filter)) * np.sqrt(rows * cols)
            ifftFilterArray[s] = ifftFilt
            # Convolve image with even and odd filters returning the result in EO
            EO[s,o] = ifft2(np.multiply(imagefft,filter))
            An = np.abs(EO[s,o])
            sumAn_ThisOrient = sumAn_ThisOrient + An
            sumE_ThisOrient = sumE_ThisOrient + real(EO[s,o])
            sumO_ThisOrient = sumO_ThisOrient + imag(EO[s,o])
            if s == 1:
                EM_n = sum(sum(filter ** 2))
                maxAn = An
            else:
                maxAn = np.amax(maxAn,An)
        # Get weighted mean filter response vector, this gives the weighted mean
# phase angle.
        XEnergy = np.sqrt(sumE_ThisOrient ** 2 + sumO_ThisOrient ** 2) + epsilon
        MeanE = sumE_ThisOrient / XEnergy
        MeanO = sumO_ThisOrient / XEnergy
        # Now calculate An(cos(phase_deviation) - | sin(phase_deviation)) | by
# using dot and cross products between the weighted mean filter response
# vector and the individual filter response vectors at each scale.  This
# quantity is phase congruency multiplied by An, which we call energy.
        for s in np.arange(1,nscale+1).reshape(-1):
            E = real(EO[s,o])
            O = imag(EO[s,o])
            # convolution results.
            Energy = Energy + np.multiply(E,MeanE) + np.multiply(O,MeanO) - np.abs(np.multiply(E,MeanO) - np.multiply(O,MeanE))
        # Compensate for noise
# We estimate the noise power from the energy squared response at the
# smallest scale.  If the noise is Gaussian the energy squared will have a
# Chi-squared 2DOF pdf.  We calculate the median energy squared response
# as this is a robust statistic.  From this we estimate the mean.
# The estimate of noise power is obtained by dividing the mean squared
# energy value by the mean squared filter value
        medianE2n = median(reshape(np.abs(EO[1,o]) ** 2,1,rows * cols))
        meanE2n = - medianE2n / np.log(0.5)
        estMeanE2n[o] = meanE2n
        noisePower = meanE2n / EM_n
        # Now estimate the total energy^2 due to noise
# Estimate for sum(An^2) + sum(Ai.*Aj.*(cphi.*cphj + sphi.*sphj))
        EstSumAn2 = zero
        for s in np.arange(1,nscale+1).reshape(-1):
            EstSumAn2 = EstSumAn2 + ifftFilterArray[s] ** 2
        EstSumAiAj = zero
        for si in np.arange(1,(nscale - 1)+1).reshape(-1):
            for sj in np.arange((si + 1),nscale+1).reshape(-1):
                EstSumAiAj = EstSumAiAj + np.multiply(ifftFilterArray[si],ifftFilterArray[sj])
        sumEstSumAn2 = sum(sum(EstSumAn2))
        sumEstSumAiAj = sum(sum(EstSumAiAj))
        EstNoiseEnergy2 = 2 * noisePower * sumEstSumAn2 + 4 * noisePower * sumEstSumAiAj
        tau = np.sqrt(EstNoiseEnergy2 / 2)
        EstNoiseEnergy = tau * np.sqrt(pi / 2)
        EstNoiseEnergySigma = np.sqrt((2 - pi / 2) * tau ** 2)
        T = EstNoiseEnergy + k * EstNoiseEnergySigma
        # The estimated noise effect calculated above is only valid for the PC_1 measure.
# The PC_2 measure does not lend itself readily to the same analysis.  However
# empirically it seems that the noise effect is overestimated roughly by a factor
# of 1.7 for the filter parameters used here.
        T = T / 1.7
        # suit the PC_2 phase congruency measure
        Energy = np.amax(Energy - T,zero)
        EnergyAll = EnergyAll + Energy
        AnAll = AnAll + sumAn_ThisOrient
    
    ResultPC = EnergyAll / AnAll
    return ResultPC
    ############################################################################
# LOWPASSFILTER - Constructs a low-pass butterworth filter.
    
    # usage: f = lowpassfilter(sze, cutoff, n)
    
    # where: sze    is a two element vector specifying the size of filter
#               to construct [rows cols].
#        cutoff is the cutoff frequency of the filter 0 - 0.5
#        n      is the order of the filter, the higher n is the sharper
#               the transition is. (n must be an integer >= 1).
#               Note that n is doubled so that it is always an even integer.
    
    #                      1
#      f =    --------------------
#                              2n
#              1.0 + (w/cutoff)
    
    # The frequency origin of the returned filter is at the corners.
    
    # See also: HIGHPASSFILTER, HIGHBOOSTFILTER, BANDPASSFILTER
    
    # Copyright (c) 1999 Peter Kovesi
# School of Computer Science & Software Engineering
# The University of Western Australia
# http://www.csse.uwa.edu.au/
    
    # Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, subject to the following conditions:
    
    # The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
    
    # The Software is provided "as is", without warranty of any kind.
    
    # October 1999
# August  2005 - Fixed up frequency ranges for odd and even sized filters
#                (previous code was a bit approximate)
    
    
def lowpassfilter(sze = None,cutoff = None,n = None): 
    if cutoff < 0 or cutoff > 0.5:
        raise Exception('cutoff frequency must be between 0 and 0.5')
    
    if rem(n,1) != 0 or n < 1:
        raise Exception('n must be an integer >= 1')
    
    if len(sze) == 1:
        rows = sze
        cols = sze
    else:
        rows = sze(1)
        cols = sze(2)
    
    # Set up X and Y matrices with ranges normalised to +/- 0.5
# The following code adjusts things appropriately for odd and even values
# of rows and columns.
    if np.mod(cols,2):
        xrange = np.array([np.arange(- (cols - 1) / 2,(cols - 1) / 2+1)]) / (cols - 1)
    else:
        xrange = np.array([np.arange(- cols / 2,(cols / 2 - 1)+1)]) / cols
    
    if np.mod(rows,2):
        yrange = np.array([np.arange(- (rows - 1) / 2,(rows - 1) / 2+1)]) / (rows - 1)
    else:
        yrange = np.array([np.arange(- rows / 2,(rows / 2 - 1)+1)]) / rows
    
    x,y = np.meshgrid(xrange,yrange)
    radius = np.sqrt(x ** 2 + y ** 2)
    
    f = ifftshift(1 / (1.0 + (radius / cutoff) ** (2 * n)))
    
    return f
    return FSIM,FSIMc