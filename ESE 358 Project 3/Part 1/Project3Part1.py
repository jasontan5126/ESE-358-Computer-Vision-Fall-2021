"""
Jason Tan
112319102
ESE 358 Project 3 Part 1
"""

#Part 1 Project 3 Histogram Equalization
import cv2
import numpy
import numpy as np
import skimage
from skimage import io
import matplotlib.pyplot as plt

def RGB_to_gray(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayImage

rgbImage = io.imread('freyhall256.jpg');   #to read the image

M = rgbImage.shape
N = rgbImage.shape

grayImage = RGB_to_gray(rgbImage)

#initialize p (discrete probability of brightness) as an 1-D array which is discrete probability
p = np.zeros(256)
#initialize c (cumulative distribution function) as a 1-D array which is a cumulative distribution function
c = np.zeros(256)
#initialize the histogram
h1 = np.zeros(256)
rgbImage1 = np.zeros((256,256), dtype=np.uint8)

for m in range(0, 255):
    for n in range(0, 255):

        #histogram array with input image (gray image)
        histogramInputImage = grayImage[m,n]
        h1[histogramInputImage] = h1[histogramInputImage] + 1
        #Compute the discrete probability for brightness

#compute the discrete probability of brightness
p = h1/(numpy.dot(M[0], N[1]))

for k in range(0, 255):
    #Cumulative distribution function
    c[k + 1] = c[k] + p[k + 1]


for m in range(0, 255):
    for n in range (0, 255):
        rgbImage1[m,n] = 255 * c[grayImage[m, n]]

plt.title('Gray level Image')
plt.imshow(grayImage, cmap=plt.cm.gray)
plt.show()

plt.title('Histogram equalization image')
plt.imshow(rgbImage1, cmap=plt.cm.gray)
cv2.imwrite('outputPart1.jpg', rgbImage1)
plt.show()