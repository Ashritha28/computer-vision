import cv2
import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt

#Read a grayscale image
def read_image(path):
    image = cv2.imread(path,0)
    return image

#Resize the input image
def resize_image(image,x,y):
    resized_image = cv2.resize(image,(x,y))
    return resized_image

#Displays an image
def display_image(image):
    plt.imshow(image)
    plt.show()

#Gaussian blurring for noise removal
def gaussian_smoothing(image,filter_x,filter_y):
    blur_image = cv2.GaussianBlur(image,(filter_x,filter_y),0)
    return blur_image

#Returns a gaussian filter with the specified kernel size and sigma
def gaussian_kernel(kernel_size,sigma):
    interval = (2*sigma+1.)/(kernel_size)
    x = np.linspace(-sigma-interval/2., sigma+interval/2., kernel_size+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def harris_corner_detector(image,kernel,sobel_filter_size,hc_constant,threshold):
    partial_der_x = cv2.Sobel(image,cv2.CV_64F,1,0,sobel_filter_size)
    partial_der_y = cv2.Sobel(image,cv2.CV_64F,0,1,sobel_filter_size)
    prod_der_xx = np.multiply(partial_der_x,partial_der_x)
    prod_der_yy = np.multiply(partial_der_y,partial_der_y)
    prod_der_xy = np.multiply(partial_der_x,partial_der_y)
    
    sum_xx = cv2.filter2D(prod_der_xx, -1, kernel)
    sum_yy = cv2.filter2D(prod_der_yy, -1, kernel)
    sum_xy = cv2.filter2D(prod_der_xy, -1, kernel)
    
    trace =  sum_xx + sum_yy
    det = (sum_xx * sum_yy)-(sum_xy**2)
    response = det - hc_constant*(trace**2)
    
    result_image = image.copy()
    result_image = cv2.cvtColor(result_image,cv2.COLOR_GRAY2RGB)
    
    corner_points = []
    rows,cols = response.shape
    for i in range(rows):
        for j in range(cols):
            if(response[i,j]>threshold*response.max()):
                corner_points.append([i,j,response[i,j]])
                result_image.itemset((i,j,0),255)
                result_image.itemset((i,j,1),0)
                result_image.itemset((i,j,2),0)

    return result_image,corner_points

def projection_transform_matrix(A,B):
    C = np.zeros((8,9))
    ones = np.ones(3)
    
    P = A[0:3]    
    P = np.c_[P,ones]
    
    Q = A[3]
    Q = np.append(Q,1)
    temp1 = np.linalg.lstsq(P, Q)[0]
    temp1= np.repeat(temp1,3).reshape(3,3).T    
    scaled_P = np.multiply(P,temp1)
    
    R = B[0:3]
    R = np.c_[R,ones]
    
    S = B[3]
    S = np.append(S,1)
    temp2 = np.linalg.lstsq(R, S)[0]
    temp2 = np.repeat(temp2,3).reshape(3,3).T
    scaled_Q = np.multiply(Q,temp2)
    
    scaled_P_inv = np.linalg.inv(scaled_P)
    H = np.dot(scaled_P_inv,scaled_Q)
    return H

def apply_transform(H,img1,img2,color_img1,color_img2):
	setX=[]; setY=[]
	p1 = np.array([[0],[0],[1]])
	p1 = np.dot(H,p1)
	setX.append(p1[0]/p1[2])
	setY.append(p1[1]/p1[2])

	p2 = np.array([[img1.shape[0]-1],[0],[1]])
	p2 = np.dot(H,p2)
	setX.append(p2[0]/p2[2])
	setY.append(p2[1]/p2[2])

	p3 = np.array([[0],[img1.shape[1]-1],[1]])
	p3 = np.dot(H,p3)
	setX.append(p3[0]/p3[2])
	setY.append(p3[1]/p3[2])

	p4 = np.array([[img1.shape[0]-1],[img1.shape[1]-1],[1]])
	p4 = np.dot(H,p4)
	setX.append(p4[0]/p4[2])
	setY.append(p4[1]/p4[2])

	l_x = int(np.min(setX))
	h_x = int(np.max(setX))
	l_y = int(np.min(setY))
	h_y = int(np.max(setY))

	res = np.zeros((int(h_x-l_x),int(h_y-l_y)+img2.shape[1],3),dtype=np.uint8)
	for i in range(img1.shape[0]):
		for j in range(img1.shape[1]):
        		point = np.array([[i],[j],[1]])
        		point = np.dot(H,point)
        		x1=int(point[0]/point[2])-l_x
        		y1=int(point[1]/point[2])-l_y

       			if x1>0 and y1>0 and y1<h_y-l_y and x1<h_x-l_x:
            			res[x1,y1]=color_img1[i,j]

	for i in range(img2.shape[0]):
    		for j in range(img2.shape[1]):
        		if np.array_equal(res[i-l_x][j-l_y],np.array([0,0,0])):
            			res[i-l_x][j-l_y]=color_img2[i][j]

	return res

def plotCommonCorners(points, image) :
    num = points.shape[0]
    for i in range(num) :
        for j in range(5):
            for k in range(5):
                image.itemset((points[i][0]+j,points[i][1]+k,0),255)
                image.itemset((points[i][0]+j,points[i][1]+k,1),0)
                image.itemset((points[i][0]+j,points[i][1]+k,2),0)
    return image