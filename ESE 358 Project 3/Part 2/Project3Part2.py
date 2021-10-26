#Part 2 Project 3 Image filtering and convolution
import cv2
import numpy
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

def RGB_to_gray(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayImage

readImagePart2 = io.imread('waterfall256.jpg')  # to read the image for part 2
grayImagePart2 = RGB_to_gray(readImagePart2)

intMatrixFilter1 = numpy.zeros((5, 5))
intMatrixFilter2 = numpy.zeros((5, 5))

#To read from the file and turn into a multidimensional array from filter 1 and turn from string into regular numbers so it doesn't
#interpret bidimensional array as a string (Think this is a gaussian filter)
#0, 0 -1, 0 ...
with open("filter1.txt") as textFile:

    filter1 = [line.split() for line in textFile]
    res = [int(i) for i in filter1[0]]
    res1 = [int(i) for i in filter1[1]]
    res2 = [int(i) for i in filter1[2]]
    res3 = [int(i) for i in filter1[3]]
    res4 = [int(i) for i in filter1[4]]

    intMatrixFilter1[0] = res
    intMatrixFilter1[1] = res1
    intMatrixFilter1[2] = res2
    intMatrixFilter1[3] = res3
    intMatrixFilter1[4] = res4

#To read from the file and turn into a multidimensional array from filter 2 and turn from string into regular numbers so it doesn't
#interpret bidimensional array as a string (Think this is a mean filter)
#0.4, 0.4, ...
with open("filter2.txt") as textFile:
    filter2 = [line.split() for line in textFile]
    res5 = [float(i) for i in filter2[0]]
    res6 = [float(i) for i in filter2[1]]
    res7 = [float(i) for i in filter2[2]]
    res8 = [float(i) for i in filter2[3]]
    res9 = [float(i) for i in filter2[4]]

    intMatrixFilter2[0] = res5
    intMatrixFilter2[1] = res6
    intMatrixFilter2[2] = res7
    intMatrixFilter2[3] = res8
    intMatrixFilter2[4] = res9


N = readImagePart2.shape[0]   #size of the input image which would be applied for the output image
outputImagePart2Filter1 = np.zeros((N, N))     #For the output image to have the size of 256 x 256 as input image which applies filter 1
outputImagePart2Filter2 = np.zeros((N, N))     #For the output image to have the size of 256 x 256 as input image which applies filter 2
rPart2Filter1 = np.zeros((N, N))   #Denoted as r(i,j) for filter 1
rPart2Filter2 = np.zeros((N, N))   #Denoted as r(i,j) for filter 2

#i, j is at range between 3 and N - 2
for i in range (3, N - 2):
    for j in range (3, N - 2):
        #k, l is for range between 1 and 5 with summation
        for k in range(1, 5):
            for l in range(1, 5):
                #Output image in the interior pixels with different filters applied
                outputImagePart2Filter1[i, j] += (intMatrixFilter1[k][l] * grayImagePart2[(i - (k - 3)), (j - (l - 3))])
                outputImagePart2Filter2[i, j] += (intMatrixFilter2[k][l] * grayImagePart2[(i - (k - 3)), (j - (l - 3))])


#Compute the output image as r(i,j) = (h(i,j)-hmin*255/(hmax-hmin)
for i in range(3, N - 2):
    for j in range(3, N - 2):
        hmin = grayImagePart2.min()
        hmax = grayImagePart2.max()
        rPart2Filter1[i, j] = ((outputImagePart2Filter1[i, j] - hmin)*255)/(hmax-hmin)
        rPart2Filter2[i, j] = ((outputImagePart2Filter2[i, j] - hmin)*255)/(hmax - hmin)

#To display all the image outputs for part 2
plt.title('Part 2: Input Image')
plt.imshow(grayImagePart2, cmap=plt.cm.gray)
plt.show()

plt.title('Part 2 Output image with filter 1 applied')
plt.imshow(rPart2Filter1, cmap=plt.cm.gray)
cv2.imwrite('part2fFilter1Output.jpg', rPart2Filter1)
plt.show()

plt.title('Part 2 Output image with filter 2 applied')
plt.imshow(rPart2Filter2, cmap=plt.cm.gray)
cv2.imwrite('part2Filter2Output.jpg', rPart2Filter2)
plt.show()