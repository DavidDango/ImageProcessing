import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm_gui


# Selecciona N numeros entre 0 y M - 1 sin repeticion.
def pickN(N, M):
	return random.sample(range(M), N)


# Retorna el numero de elementos de destination que tras la transformacion T se encuentran dentro un radio threshold de su equivalente en source.
def findPermisible(source, destination, T, threshold):
	destination_p = np.dot(destination, T)
	p = 0
	for i in range(len(source)):
		if np.linalg.norm(destination_p[i] - source[i]) < threshold:
			p = p + 1
	return p


# Metodo RANSAC.
def RANSAC(source, destination, maxiter, threshold, ratio):
	iter = 0
	best_T = None
	best_per = -1
	if len(source) != len(destination):
		raise Exception("Different length in source and destination points")
	N = 3
	M = len(source)
	while iter < maxiter:
		points = pickN(N, M)
		x = []
		y = []
		for i in points:
			x.append(source[i])
			y.append(destination[i])
		T = np.linalg.solve(np.array(y), np.array(x))
		per = findPermisible(source, destination, T, threshold)
		if per > M*ratio:
			return T
		elif per > best_per:
			best_per = per
			best_T = T
		iter = iter + 1 
	return best_T


# Calcula el color correspondiente despues de la interpolacion bilineal dentro de imagen original.
def computeColour(image, x, y, a, b):
	colour = np.array([0, 0, 0])
	if x >= 0 and y >= 0 and x < image.shape[0] and y < image.shape[1]:
		colour = colour + (image[x][y]*(1 - a)*(1 - b))
	if x + 1 >= 0 and y >= 0 and x + 1 < image.shape[0] and y < image.shape[1]:
		colour = colour + (image[x + 1][y]*(1 - b)*a)
	if x >= 0 and y + 1 >= 0 and x < image.shape[0] and y + 1 < image.shape[1]:
		colour = colour + (image[x][y + 1]*(1 - a)*b)
	if x + 1 >= 0 and y + 1 >= 0 and x + 1 < image.shape[0] and y + 1 < image.shape[1]:
		colour = colour + (image[x + 1][y + 1]*b*a)
	return colour

sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=6, sigma = 1.6)

# Imagen original.
img = cv2.imread('./casos/caso_1/1a.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
kp = sift.detect(gray)
kp, des = sift.compute(gray, kp)

# Imagen a agregar.
img_2 = cv2.imread('./casos/caso_1/1b.jpg')
gray_2= cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)
kp_2 = sift.detect(gray_2)
kp_2, des_2 = sift.compute(gray_2, kp_2)

# Matching
matcher=cv2.BFMatcher()
matches = matcher.knnMatch(des,des_2, k=2)
good = []
for m,n in matches:
    if m.distance < 0.8*n.distance:
        good.append(m)

src_pts_z = []
dst_pts_z = []

for m in good:
	x = kp[m.queryIdx].pt
	y = kp_2[m.trainIdx].pt
	if not (x + (1,)) in src_pts_z and not (y + (1,)) in dst_pts_z:
		src_pts_z.append(x + (1,))
		dst_pts_z.append(y + (1,))

src_pts_z = np.float32(src_pts_z)
dst_pts_z = np.float32(dst_pts_z)

T = RANSAC(src_pts_z, dst_pts_z, 10000, 4, 0.6)

# Puntos extremos.
ul = np.dot([0, 0, 1], T)
ur = np.dot([img_2.shape[1], 0, 1], T)
ll = np.dot([0, img_2.shape[0], 1], T)
lr = np.dot([img_2.shape[1], img_2.shape[0], 1], T)

ogul = [0, 0, 1]
ogur = [img.shape[1], 0, 1]
ogll = [0, img.shape[0], 1]
oglr = [img.shape[1], img.shape[0], 1]

minx = int(min([0, ul[0], ur[0], ll[0], lr[0]]))
maxx = int(max([ogur[0], ul[0], ur[0], ll[0], lr[0]]))

miny = int(min([0, ul[1], ll[1], ur[1], lr[1]]))
maxy = int(max([ogll[1], ul[1], ll[1], ur[1], lr[1]]))

#Largo y alto de imagen.
y_l = maxx - minx
x_l = maxy - miny

picture = []

T_inv = np.linalg.inv(T) 

# Generacion de nueva imagen.
for i in tqdm_gui(range(x_l)):
	line = []
	for j in range(y_l):
		[y_inv, x_inv, z] = np.dot([j + minx, i + miny, 1], T_inv)
		q = int(x_inv)
		p = int(y_inv)
		a = y_inv - p
		b = x_inv - q
		flag = False
		if i + miny >= 0 and j + minx >= 0 and i + miny < oglr[1] and j + minx < oglr[0]:
			colour = img[i + miny][j + minx]
			flag = True
		if flag:
			colourB = computeColour(img_2, q, p, a, b)
			if colourB[0] != 0 and colourB[1] != 0 and colourB[2] != 0:
				colour = (colour + colourB)/2
		else:
			colour = computeColour(img_2, q, p, a, b)
		line.append(colour)
	picture.append(np.array(line))

picture = np.array(picture)
cv2.imwrite('panoramica.png', picture)
