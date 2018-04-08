#coding:utf-8
'''Canny Edge Detection is a multi-stage algorithm with the following stages:

1. Apply Gaussian filter to smooth the image in order to remove the noise
2. Find the intensity gradients of the image
3. Apply non-maximum suppression to get rid of spurious response to edge detection
4. Apply double threshold to determine potential edges
5. Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.

'''
#Importing the required modules
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.ndimage import sobel
from scipy.misc import imsave

#Input image path
path = "input/building.jpg"

#Reading the image in grayscale
def read_image(path): 
    image = cv2.imread(path,0)
    return image

#Display grayscale image
def display_image(image):
    plt.imshow(image,cmap="gray")
    plt.show()

'''Smoothing for noise removal can be achieved through Gaussian Blurring.

Inputs - image, width of kernel (filtersize_x), height of kernel (filtersize_y)
Width and height of the kernel should be positive and odd.
Output - Filtered image
'''
#Note - The standard deviation in X and Y direction is calculated from the kernel size as zero is passed in the GaussianBlur() function.

def gaussian_blurring(image,filtersize_x,filtersize_y):
    blur_image = cv2.GaussianBlur(image,(filtersize_x,filtersize_y),0)
    return blur_image

'''This function converts angles in radians to degrees and then approximates it to the nearest direction angle.

Input - Angle in radians
Output - Angle in one of the four possible directions (Horizontal,Vertical and 2 diagonal directions) '''
def rounding_angle(angle):
    angle = np.rad2deg(angle) % 180
    if (0 <= angle < 22.5) or (157.5 <= angle < 180):
        angle_in_deg = 0 #horizontal direction
    elif (22.5 <= angle < 67.5):
        angle_in_deg = 45 #diagonalDirection-1
    elif (67.5 <= angle < 112.5):
        angle = 90 #vertical direction
    elif (112.5 <= angle < 157.5):
        angle = 135 #diagonalDirection-2
    return angle


'''This function detects horizontal,vertical and diagonal edges in an image by calculating the edge gradient and 
direction. The edge direction angle is rounded to one of the four angles mentioned above.

Input - Smoothened image, width of Sobel filter (filtersize_x), height of Sobel filter(filtersize_y)
Output - Magnitude of gradient and direction of the edge
'''
def gradient_intensity(image,filtersize_x,filtersize_y):
    image_x = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=filtersize_x)
    image_y = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=filtersize_y)
    gradient = np.hypot(image_x, image_y) #calculating hypotenuse
    direction = np.arctan2(image_y,image_x)
    rows,cols = direction.shape
    rounded_direction = np.zeros((rows,cols), dtype = np.int32)
    
    for i in range(rows):
        for j in range(cols):
            rounded_direction[i,j] = rounding_angle(direction[i,j])
    
    return (gradient,rounded_direction)

'''Non-maximum suppression is an edge thinning technique. Non-maximum suppression can help to suppress all 
the gradient values (by setting them to 0) except the local maxima, which indicate locations with the 
sharpest change of intensity value. At every pixel, it suppresses the edge strength of the center pixel 
(by setting its value to 0) if its magnitude is not greater than the magnitude of the two neighbors in the 
gradient direction.

Input - ndarray that contains the Gradient magnitude values of all the pixels, direction of the gradient 
at each pixel
Output - ndarray that constitutes only the edge pixels and zero in all other places
'''
def non_max_suppression(gradient,direction):
    rows,cols = direction.shape
    result_image = np.zeros((rows,cols),dtype = np.int32)
    for i in range(rows):
        for j in range(cols):
            if(direction[i,j]==0):
                if (j!=0 and gradient[i,j]>=gradient[i,j-1]) and (j!=cols-1 and gradient[i,j]>=gradient[i,j+1]):
                    result_image[i,j]=gradient[i,j]
            elif (direction[i,j]==90):
                if (i!=0 and gradient[i,j]>=gradient[i-1,j]) and (i!=rows-1 and gradient[i,j]>=gradient[i+1,j]):
                    result_image[i,j]=gradient[i,j]
            elif(direction[i,j]==45):
                if (i!=0 and j!=cols-1 and gradient[i,j]>=gradient[i-1,j+1]) and (i!=rows-1 and j!=0 and gradient[i,j]>=gradient[i+1,j-1]):
                    result_image[i,j]=gradient[i,j]
            elif(direction[i,j]==135):
                if (i!=0 and j!=0 and gradient[i,j] >= gradient[i-1,j-1]) and (i!=rows-1 and j!=cols-1 and gradient[i,j] >= gradient[i+1,j+1]):
                        result_image[i,j] = gradient[i,j]
    return result_image

'''Double thresholding helps in avoiding spurious edges by filtering out pixels with a weak gradient 
value and preserving edge pixels with a high gradient value. This is accomplished by selecting high and 
low threshold values. If an edge pixel’s gradient value is higher than the high threshold value, 
it is marked as a strong edge pixel. If an edge pixel’s gradient value is smaller than the high threshold 
value and larger than the low threshold value, it is marked as a weak edge pixel. If an edge pixel's value is 
smaller than the low threshold value, it will be suppressed.

Input - Image after non-maximum suppression, low threshold, high threshold
Output - Thresholded image
'''
def double_thresholding(gradient,low_thresh,high_thresh):
    cf = {
        'WEAK': np.int32(128),
        'STRONG': np.int32(255),
        'ZERO': np.int32(0),
    }
    strong_i, strong_j = np.where(gradient > high_thresh)
    weak_i, weak_j = np.where((gradient >= low_thresh) & (gradient <= high_thresh))
    zero_i, zero_j = np.where(gradient < low_thresh)
    
    gradient[strong_i, strong_j] = cf.get('STRONG')
    gradient[weak_i, weak_j] = cf.get('WEAK')
    gradient[zero_i, zero_j] = np.int32(0)
    
    return (gradient,cf.get('WEAK')) 

'''Usually a weak edge pixel caused from true edges will be connected to a strong edge pixel while noise responses are 
unconnected. To track the edge connection, blob analysis is applied by looking at a weak edge pixel and its 
8-connected neighborhood pixels. As long as there is one strong edge pixel that is involved in the blob, 
that weak edge point can be identified as one that should be preserved.

Input - Thresholded image, weak threshold, strong threshold
Output - Final image after edge detection using Canny's algorithm
'''

def hysteresis_tracking(gradient,weak,strong):
    rows,cols = gradient.shape
    for i in range(rows):
        for j in range(cols):
            if(gradient[i,j]== weak):
                if((i!=rows-1 and gradient[i+1,j]== strong) or (i!=0 and gradient[i-1,j]== strong)
                   or (j!=cols-1 and gradient[i,j+1]== strong) or (j!=0 and gradient[i,j-1]== strong)
                   or (i!=rows-1 and j!=cols-1 and gradient[i+1,j+1]== strong) 
                   or (i!=0 and j!=0 and gradient[i-1,j-1]== strong)):
                    gradient[i,j]= strong
                else:
                    gradient[i,j]=0
        return gradient

'''This function assigns colors to the edge pixels. The intensity of color depends on the magnitude 
and direction of gradient at the particular pixel.

Input - edge image, Gradient magnitude, direction of the gradient 
at each pixel
Output - HSV colored image 
'''
def hsv_color(image,gradient,direction):
    rows,cols = image.shape
    result_hsv_image = np.zeros((rows,cols,3), dtype = np.uint8)
    max_gradient = np.max(gradient)
    min_gradient = np.min(gradient)
    for i in range(rows):
        for j in range(cols):
            if(image[i][j]) :
                v = int(255*((gradient[i][j] - min_gradient)/(max_gradient - min_gradient)))
                if(direction[i][j] == 0) :
                    result_hsv_image[i][j] = [0,255,v]
                elif(direction[i][j] == 45) :
                     result_hsv_image[i][j] = [45,255,v]
                elif(direction[i][j] == 90) :
                     result_hsv_image[i][j] = [90,255,v]
                else :
                     result_hsv_image[i][j] = [135,255,v]
    
    result_hsv_image = cv2.cvtColor(result_hsv_image,cv2.COLOR_HSV2RGB)           
    return result_hsv_image

'''
Canny edge detector function that calls all the above functions in the same order.
Input - Image, width of Gaussian kernel, height of Gaussian kernel, width of Sobel kernel, height of Sobel 
kernel, low threshold, high threshold 
'''
def canny_edge_detector(image,gf_x,gf_y,sf_x,sf_y,low,high):
    smooth_image = gaussian_blurring(image,gf_x,gf_y)
    gradient,direction = gradient_intensity(smooth_image,sf_x,sf_y)
    supress_image = non_max_suppression(gradient,direction)
    threshold_image,weak = double_thresholding(supress_image,low,high)
    final_image = hysteresis_tracking(threshold_image,weak,255)
    return final_image

input_image = read_image(path)
smooth_image = gaussian_blurring(input_image,5,5)
gradient,direction = gradient_intensity(smooth_image,3,3)
edge_image = canny_edge_detector(input_image,5,5,3,3,50,100)
hsv_image = hsv_color(edge_image,gradient,direction)
display_image(hsv_image)
#Saves the image files in output folder
plt.imsave('output/result.jpeg',hsv_image)
