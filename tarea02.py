import sys
sys.path.append('./pai_basis')
import basis
import pai_io
import orientation_histograms as oh
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy.ndimage.filters as nd_filters
import math

from scipy.ndimage import gaussian_filter
import cv2


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
    # image = gaussian_filter(image, sigma=1.5)
    h = np.zeros(k , np.float32)
    gx_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    gy_mask = np.transpose(gx_mask)
    gx = nd_filters.convolve(image.astype(np.float32), gx_mask)
    gy = nd_filters.convolve(image.astype(np.float32), gy_mask)
    Lx = np.zeros((B, B) , np.float32)
    Ly = np.zeros((B, B) , np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            Lx[int(i*B/image.shape[0])][int(j*B/image.shape[1])] = Lx[int(i*B/image.shape[0])][int(j*B/image.shape[1])] + (math.pow(gx[i][j], 2) - math.pow(gy[i][j], 2))
            Ly[int(i*B/image.shape[0])][int(j*B/image.shape[1])] = Ly[int(i*B/image.shape[0])][int(j*B/image.shape[1])] + 2*gx[i][j]*gy[i][j]
    # Lx = gaussian_filter(Lx, sigma=0.5)
    # Ly = gaussian_filter(Ly, sigma=0.5)
    a = Ly/Lx
    a = (np.arctan(L) - np.pi)/2
    a[a < 0] = a[a < 0] + np.pi
    mag = np.sqrt(np.square(Lx) + np.square(Ly))
    indx = np.round(k*a/np.pi)
    indx[indx == k] = 0
    for i in a:
        rows, cols = np.where(indx  == i)
        h[i] = np.sum(mag[rows, cols])
    h =  h / np.linalg.norm(h,2)
    return h


def histogram_shelo(image, B, k):
    # image = gaussian_filter(image, sigma=1.5)
    h = np.zeros(k , np.float32)
    pixels = []

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            temp2 = []
            p = i*B/image.shape[0]
            q = j*B/image.shape[1]
            l_pos = int(p - 0.5)
            n_pos = int(q - 0.5)
            r_pos = int(p + 0.5)
            s_pos = int(q + 0.5)
            if l_pos < 0:
                l_pos = 0
            if n_pos < 0:
                l_pos = 0
            if r_pos >= B:
                r_pos = B - 1
            if s_pos >= B:
                s_pos = B - 1
            dist = p - int(p)
            if dist < 0.5:
                l_weight = 0.5 - dist
                r_weight = 1 - l_weight
            else:
                r_weight = dist - 0.5
                l_weight = 1 - r_weight
            dist = q - int(q)
            if dist < 0.5:
                n_weight = 0.5 - dist
                s_weight = 1 - s_weight
            else:
                s_weight = dist - 0.5
                n_weight = 1 - n_weight
            temp2.append([[l_pos, n_pos, l_weight, n_weight], [r_pos, n_pos, r_weight, n_weight], [l_pos, s_pos, l_weight, s_weight], [r_pos, s_pos, r_weight, s_weight]])
        pixels.append(temp2)

    gx_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    gy_mask = np.transpose(gx_mask)
    gx = nd_filters.convolve(image.astype(np.float32), gx_mask)
    gy = nd_filters.convolve(image.astype(np.float32), gy_mask)
    A = np.zeros((B, B) , np.float32)
    D = np.zeros((B, B) , np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for ar in pixels[i][j]:
                cx = ar[0]
                cy = ar[1]
                a = ar[2]*ar[3]
                if cx >= 0 and cx < image.shape[0] and cy >= 0 and cy < image.shape[1]:
                    A[cx][cy] = A[cx][cy] + ((math.pow(gx[i][j], 2) - math.pow(gy[i][j], 2)))*a
                    D[cx][cy] = D[cx][cy] + 2*gx[i][j]*gy[i][j]*a

    # A = gaussian_filter(A, sigma=0.5)
    # D = gaussian_filter(D, sigma=0.5)
    b = A/D
    b = (np.arctan(L) - np.pi)/2
    b[b < 0] = b[b < 0] + np.pi
    mag = np.sqrt(np.square(Lx) + np.square(Ly))
    indx = np.round(k*a/np.pi)
    indx[indx == k] = 0
    for i in a:
        rows, cols = np.where(indx  == i)
        h[i] = np.sum(mag[rows, cols])
    h =  h / np.linalg.norm(h,2)
    return h


def overlapped_helo(image, B, k):
    # image = gaussian_filter(image, sigma=1.5)
    gx_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    gy_mask = np.transpose(gx_mask)
    gx = nd_filters.convolve(image.astype(np.float32), gx_mask)
    gy = nd_filters.convolve(image.astype(np.float32), gy_mask)
    Lx = np.zeros((B, B) , np.float32)
    Ly = np.zeros((B, B) , np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            Lx[int(i*B/image.shape[0])][int(j*B/image.shape[1])] = Lx[int(i*B/image.shape[0])][int(j*B/image.shape[1])] + (math.pow(gx[i][j], 2) - math.pow(gy[i][j], 2))
            Ly[int(i*B/image.shape[0])][int(j*B/image.shape[1])] = Ly[int(i*B/image.shape[0])][int(j*B/image.shape[1])] + 2*gx[i][j]*gy[i][j]
    
    # Lx = gaussian_filter(Lx, sigma=0.5)
    # Ly = gaussian_filter(Ly, sigma=0.5)

    angles = np.arctan2(Ly, Lx)/2
    mag = np.sqrt(np.square(Lx) + np.square(Ly))

    max_mag = np.amax(mag)

    fig, xs = plt.subplots(1,2)
    for i in range(2):
        xs[i].set_axis_off()
    xs[0].imshow(image, cmap = 'gray')
    xs[0].set_title('Image')
    xs[1].imshow(image, cmap = 'gray')
    xs[1].set_title('Orientations')
    dx = image.shape[0]/B
    dy = image.shape[1]/B
    for i in range(B):
        for j in range(B):
            x1 = dx*(i + 1/2 + math.cos(-angles[i][j])*mag[i][j]/(max_mag*2))
            x2 = dx*(i + 1/2 - math.cos(-angles[i][j])*mag[i][j]/(max_mag*2))
            y1 = dy*(j + 1/2 + math.sin(-angles[i][j])*mag[i][j]/(max_mag*2))
            y2 = dy*(j + 1/2 - math.sin(-angles[i][j])*mag[i][j]/(max_mag*2))
            xs[1].plot([y1, y2], [x1, x2])
    plt.show()


def overlapped_shelo(image, B, k):
    pixels = []
    for i in range(image.shape[0]):
        temp2 = []
        for j in range(image.shape[1]):
            p = i*B/image.shape[0]
            q = j*B/image.shape[1]
            l_pos = int(p - 0.5)
            n_pos = int(q - 0.5)
            r_pos = int(p + 0.5)
            s_pos = int(q + 0.5)
            if l_pos < 0:
                l_pos = 0
            if n_pos < 0:
                l_pos = 0
            if r_pos >= B:
                r_pos = B - 1
            if s_pos >= B:
                s_pos = B - 1
            dist = p - int(p)
            if dist < 0.5:
                l_weight = 0.5 - dist
                r_weight = 1 - l_weight
            else:
                r_weight = dist - 0.5
                l_weight = 1 - r_weight
            dist = q - int(q)
            if dist < 0.5:
                n_weight = 0.5 - dist
                s_weight = 1 - n_weight
            else:
                s_weight = dist - 0.5
                n_weight = 1 - s_weight
            temp2.append([[l_pos, n_pos, l_weight, n_weight], [r_pos, n_pos, r_weight, n_weight], [l_pos, s_pos, l_weight, s_weight], [r_pos, s_pos, r_weight, s_weight]])
        pixels.append(temp2)

    # Lx = gaussian_filter(Lx, sigma=0.5)
    # Ly = gaussian_filter(Ly, sigma=0.5)

    gx_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    gy_mask = np.transpose(gx_mask)
    gx = nd_filters.convolve(image.astype(np.float32), gx_mask)
    gy = nd_filters.convolve(image.astype(np.float32), gy_mask)
    A = np.zeros((B, B) , np.float32)
    D = np.zeros((B, B) , np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for ar in pixels[i][j]:
                cx = ar[0]
                cy = ar[1]
                a = ar[2]*ar[3]
                if cx >= 0 and cx < image.shape[0] and cy >= 0 and cy < image.shape[1]:
                    A[cx][cy] = A[cx][cy] + ((math.pow(gx[i][j], 2) - math.pow(gy[i][j], 2)))*a
                    D[cx][cy] = D[cx][cy] + 2*gx[i][j]*gy[i][j]*a


    angles = np.arctan2(D, A)/2
    mag = np.sqrt(np.square(D) + np.square(A))

    max_mag = np.amax(mag)

    fig, xs = plt.subplots(1,2)
    for i in range(2):
        xs[i].set_axis_off()
    xs[0].imshow(image, cmap = 'gray')
    xs[0].set_title('Image')
    xs[1].imshow(image, cmap = 'gray')
    xs[1].set_title('Orientations')
    dx = image.shape[0]/B
    dy = image.shape[1]/B
    for i in range(B):
        for j in range(B):
            x1 = dx*(i + 1/2 + math.cos(-angles[i][j])/2)
            x2 = dx*(i + 1/2 - math.cos(-angles[i][j])/2)
            y1 = dy*(j + 1/2 + math.sin(-angles[i][j])/2)
            y2 = dy*(j + 1/2 - math.sin(-angles[i][j])/2)
            xs[1].plot([y1, y2], [x1, x2], 'y')
    plt.show()


filename = './dataset_1/BD_2/chair/180016.jpg'
# filename = 'Y-FlyerLogo.png'
image = pai_io.imread(filename, as_gray = True)

# image = cv2.Canny(image, 100, 200)

overlapped_shelo(image, 25, 36)