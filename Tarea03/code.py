import cv2
import numpy as np

sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=6, sigma = 1.6)

img = cv2.imread('A.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
kp = sift.detect(gray)
kp, des = sift.compute(gray, kp)


img_2 = cv2.imread('A_Warped.jpg')
gray_2= cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)
kp_2 = sift.detect(gray_2)
kp_2, des_2 = sift.compute(gray_2, kp_2)

    
matcher=cv2.BFMatcher()
matches = matcher.knnMatch(des,des_2, k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.8*n.distance:
        good.append(m)

##Calculando homografía para afinar correspondencias
if len(good)>4:    
    src_pts = np.float32([ kp[m.queryIdx].pt for m in good ]).reshape(-1,2)
    dst_pts = np.float32([ kp_2[m.trainIdx].pt for m in good ]).reshape(-1,2)
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)
    matchesMask = mask.ravel().tolist()
else:
    matchesMask = None

eq = []

for mat in good:

    img1_idx = mat.queryIdx
    img2_idx = mat.trainIdx

    (x1,y1) = kp[img1_idx].pt
    (x2,y2) = kp_2[img2_idx].pt

    if not (x1, y1, x2, y2) in eq:
    	eq.append((x1, y1, x2, y2))

#Aquí, matchesMask contiene las correspondencias
img3 = cv2.drawMatches(img,kp,img_2,kp_2, good, None, flags=2, matchesMask=matchesMask)

imS = cv2.resize(img3, (1360, 768))

cv2.imshow("matches", imS)
cv2.waitKey()