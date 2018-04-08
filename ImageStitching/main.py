import cv2
import numpy as np
from matplotlib import pyplot as plt
from utilities import *

img1_points = np.array([[397.5,7.5],[324,276],[649.5,148.5],[505.5,25.5]])
img2_points = np.array([[426,265.5],[306,502.5],[651,436.5],[523.5,298.5]])
img1 = cv2.imread('img1.jpg',0)
img2 = cv2.imread('img2.jpg',0)

col_img1 = cv2.imread('img1.jpg')
col_img2 = cv2.imread('img2.jpg')
col_img1 = cv2.resize(col_img1,(600,800))
col_img2 = cv2.resize(col_img2,(600,800))

resized_img1 = cv2.resize(img1,(600,800))
resized_img2 = cv2.resize(img2,(600,800))

mat = projection_transform_matrix(img1_points,img2_points)
res_img = apply_transform(mat,img1,img2,col_img1,col_img2)

display_image(res_img)
plt.imsave('final.png',res_img)

all_corner_points1 = np.array([[120,249],[140,267],[165,202],[165,328],[310,195],[255,325],[337,319],[426,327],[522,322],[625,324],[568,433],[565,499],[565,520],[541,520],[451,516],[390,511],[154,513],[114,340],[462,394],[186,405]])
all_corner_points2 = np.array([[85,75],[105,97],[124,18],[145,153],[171,138],[235,145],[321,138],[412,132],[508,124],[616,115],[561,235],[559,295],[558,315],[534,315],[447,318],[390,321],[163,337],[94,169],[391,274],[177,237]])

plotted1 = plotCommonCorners(all_corner_points1, color_img1)
plt.imshow(plotted1)
plt.show()
plt.imsave('plotted1.png',plotted1)

plotted2 = plotCommonCorners(all_corner_points2, color_img2)
plt.imshow(plotted2)
plt.show()
plt.imsave('plotted2.png',plotted2)

perspective_img = cv2.imread('perspective.png')
h1 = color_img1.shape[0]
w1 = color_img1.shape[1]
projected_points = np.zeros(all_corner_points1.shape,dtype=np.uint8)
for i in range(len(all_corner_points1)):
    x,y = all_corner_points1[i]
    p = np.array([[x],[y],[1]])
    p = np.dot(H,p)
    x1 = int(p[0]/p[2])-min_x
    y1 = int(p[1]/p[2])-min_y
    projected_points[i] = np.array([x1,y1],dtype=np.uint8)

print(projected_points)
plotted_perspective = plotCommonCorners(projected_points,perspective_img)
plt.imshow(plotted_perspective)
plt.show()
plt.imsave('plotted_perspective.png',plotted_perspective)
