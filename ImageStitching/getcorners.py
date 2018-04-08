import cv2
import numpy as np
from matplotlib import pyplot as plt
from utilities import *

img1 = cv2.imread('img1.jpg',0)
img2 = cv2.imread('img2.jpg',0)

col_img1 = cv2.imread('img1.jpg')
col_img2 = cv2.imread('img2.jpg')

resized_img1 = cv2.resize(img1,(600,800))
resized_img2 = cv2.resize(img2,(600,800))

denoised_img1 = gaussian_smoothing(resized_img1,3,3)
denoised_img2 = gaussian_smoothing(resized_img2,3,3)

kernel = gaussian_kernel(5,1)
corner_img1,p1 = harris_corner_detector(denoised_img1,kernel,3,0.04,0.025)
corner_img2,p2 = harris_corner_detector(denoised_img2,kernel,3,0.04,0.025)

print(len(p1))
print(len(p2))
plt.imsave('corner_img1.png',corner_img1)
plt.imsave('corner_img2.png',corner_img2)

f = open( 'corners1.txt', 'w' )
for i in range(len(p1)):
    f.write( repr(p1[i])+'\n' )
f.close()

f = open( 'corners2.txt', 'w' )
for i in range(len(p2)):
    f.write( repr(p2[i])+'\n' )
f.close()