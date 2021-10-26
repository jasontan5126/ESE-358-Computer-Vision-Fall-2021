#Part 3 Project 3 Edge Detection

import cv2
import numpy
import numpy as np
from skimage import io
import math
import matplotlib.pyplot as plt

def RGB_to_gray(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayImage

inputImagePart3 = io.imread('waterfall256.jpg')       #Input image for part 3
grayLevelImagePart3 = RGB_to_gray(inputImagePart3)   #To turn the original input image for part 3 into a gray level image

N = grayLevelImagePart3.shape[0]   #To get the size of the gray level image
M = grayLevelImagePart3.shape[0]

outputImagePart3 = numpy.zeros((N, N))    #Initialize the output image for part 3
outputImagePart3_1 = numpy.zeros((N,N))
outputImagePart3_2 = numpy.zeros((N,N))
g = numpy.zeros(N)

#First compute the filter g(k) as:

sigma = 0.5
sum = 0;
threshold = 15

#To compute the separable Gaussian filter denoted by g(k)
for k in range(0, 8):
    g[k] = math.exp(-((k-5)*(k-5))/(2*sigma*sigma))
    sum += g[k]

#To normalize the filter coefficients as:
for k in range (0, 8):
    g[k] = g[k]/ sum

#Filter each row i
for i in range (0, M - 1):
    for j in range(5, N-6):
        sum = 0;
        for k in range(0, 8):
            sum = sum + g[k]*grayLevelImagePart3[i, j-(k-5)]
        outputImagePart3[i, j] = sum
        #Gx = grayLevelImagePart3[i, j + 1] - grayLevelImagePart3[i, j]


#Filter each column j
for j in range(0, N - 1):
    for i in range(5, M - 6):
        sum = 0
        for k in range(0, 8):
            sum = sum + g[k]*outputImagePart3[i-(k-5), j]
        outputImagePart3_1[i, j] = sum
        outputImagePart3_2[i, j] = sum


        #Gy = grayLevelImagePart3[i, j + 1] - grayLevelImagePart3[i, j]

for row in range(1, 254):
    for column in range(1, 254):
        #Gradient along rows
        gradientRowX = outputImagePart3_1[row, column + 1] - outputImagePart3_1[row, column]

        #Gradient along columns
        gradientRowY = outputImagePart3_1[row + 1, column] - outputImagePart3_1[row, column]
        gradientMagnitude = np.sqrt((np.square(gradientRowX)) + (np.square(gradientRowY)))

        # Check threshold with gradientMagnitude and set pixel to 255 or 0 for the final output image
        if gradientMagnitude > threshold:
            outputImagePart3_1[row, column] = 255

        else:
           outputImagePart3_1[row, column] = 0

plt.title('Gray level image input Part 3: Edge Detection')
plt.imshow(grayLevelImagePart3, cmap=plt.cm.gray)
plt.show()

plt.title('Gray level image with Gaussian filter applied Part 3: Edge Detection')
plt.imshow(outputImagePart3_2, cmap=plt.cm.gray)
cv2.imwrite('part3GaussianFilterOutput.jpg', outputImagePart3_2)
plt.show()

plt.title('Gray level image with edge detection and Gaussian filter applied')
plt.imshow(outputImagePart3_1, cmap=plt.cm.gray)
cv2.imwrite('part3GaussianEdgeDetectionOutput.jpg', outputImagePart3_1)
plt.show()