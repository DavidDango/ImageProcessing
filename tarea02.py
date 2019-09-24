import sys
sys.path.append('./pai_basis')
import basis
import pai_io
import orientation_histograms as oh
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

def histogram_ho(image, k):
    h = np.zeros(k , np.float32)
    gx_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    gy_mask = np.transpose(gx_mask)
    gx = nd_filters.convolve(image.astype(np.float32), gx_mask)
    gy = nd_filters.convolve(image.astype(np.float32), gy_mask)
    ang = np.arctan2(gy,gx)    
    ang[ang < 0] = ang[ang < 0] + np.pi #sin signo
    mag = np.sqrt(np.square(gy) + np.square(gx))    
    indx = np.round(k * ang / np.pi) 
    indx[indx ==  k] = 0
    for i in range(k):
        rows, cols = np.where(indx  == i)        
        h[i] = np.sum(mag[rows, cols])
    h =  h / np.linalg.norm(h,2)  #vector unitario    
    return h

def histogram_helo(image, B, k):
    image = gaussian_filter(image, sigma=1.5)
    gx_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    gy_mask = np.transpose(gx_mask)
    gx = nd_filters.convolve(image.astype(np.float32), gx_mask)
    gy = nd_filters.convolve(image.astype(np.float32), gy_mask)
    Lx = np.zeros((B, B) , np.float32)
    Ly = np.zeros((B, B) , np.float32)
    Sx = image.shape()[0]/(B*2)
    Sy = image.shape()[1]/(B*2)
    for i in range(image.shape()[0]):
        for j in range(image.shape()[1]):
            Dx = 0
            if int(i/B) < np.round(i/B):
                Dx



histogram_helo([[1, 1, 1], [1, 1, 1], [1, 1, 1]], 6, 6)