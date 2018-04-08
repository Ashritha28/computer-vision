#Importing modules
import cv2
import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt

path = "input/building.jpg"

#Read a grayscale image
def read_image(path):
    image = cv2.imread(path,0)
    return image

#Displays an image
def display_image(image):
    plt.imshow(image)
    plt.show()

#Returns a gaussian filter with the specified kernel size and sigma
def gaussian_kernel(kernel_size,sigma):
    interval = (2*sigma+1.)/(kernel_size)
    x = np.linspace(-sigma-interval/2., sigma+interval/2., kernel_size+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

'''
Parameters :
image - Image whose corners are to be detected
kernel - gaussian kernel
sobel_filter_size - Kernel size for the Sobel filter
hc_constant - Harris corner constant which can lie between 0.04 and 0.06
threshold - Threshold for the response 
'''
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

input_image = read_image(path)
kernel = gaussian_kernel(5,1)
output_image,num_corner_points = harris_corner_detector(input_image,kernel,3,0.06,0.05)

display_image(output_image)
print(len(num_corner_points))
plt.imsave("output/harris_5_3_0.06_0.05.png",output_image)
