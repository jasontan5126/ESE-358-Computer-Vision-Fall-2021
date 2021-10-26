"""
ESE 358 Project 3: Part 4a and 4b:Corner edge detection and local feature descriptor
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from skimage import io



def my_harris(window_size, k, threshold):
    img = io.imread('chessboard.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    M = gray.shape[0];
    N = gray.shape[1];

    g = np.zeros(N)
    outputImagePart4 = np.zeros((M, M))

    img_gaussian = cv2.GaussianBlur(gray, (9, 9), 2)

    height = img.shape[0]  # .shape[0] outputs height
    width = img.shape[1]  # .shape[1] outputs width .shape[2] outputs color channels of image
    matrix_R = np.zeros((height, width))

    offset = int(window_size)


    Ix = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0)
    Iy = cv2.Sobel(img_gaussian, cv2.CV_64F, 0, 1)

    A = np.square(Ix)
    B = np.square(Iy)
    C = Ix * Iy

    Aprime = cv2.GaussianBlur(A, (11, 11), 5.5)
    Bprime = cv2.GaussianBlur(B, (11, 11), 5.5)
    Cprime = cv2.GaussianBlur(C, (11, 11), 5.5)

    print("Finding Corners...")
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            H = np.array([[Aprime[y,x],Cprime[y,x]],[Cprime[y,x],Bprime[y,x]]])

            det = np.linalg.det(H)
            tr = np.matrix.trace(H)
            R = det - k * (tr ** 2)
            matrix_R[y - offset, x - offset] = R

    cv2.normalize(matrix_R, matrix_R, 0, 1, cv2.NORM_MINMAX)

    sum4 = 0
    sum5 = 0
    sum6 = 0
    sum7 = 0
    sum8 = 0
    sum9 = 0
    sum10 = 0
    sum11 = 0
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            value = matrix_R[y, x]
            if value > threshold:
               cv2.circle(img, (x, y), 1, (255, 255, 255))
               print('Pixel Coordinates of Corner pixels: ', (x, y))

               mag, ang = cv2.cartToPolar(Ix[x,y], Iy[x,y])

               for ix, iy in np.ndindex(ang.shape):
                   nearest45Degrees = round(math.degrees(ang[ix, iy]) / 45) * 45
                   magnitude = mag[ix, iy]
                   if nearest45Degrees == 0:
                       sum4 += magnitude
                   elif nearest45Degrees == 45:
                       sum5 += magnitude
                   elif nearest45Degrees == 90:
                       sum6 += magnitude
                   elif nearest45Degrees == 135:
                       sum7 += magnitude
                   elif nearest45Degrees == 180:
                       sum8 += magnitude
                   elif nearest45Degrees == 225:
                       sum9 += magnitude
                   elif nearest45Degrees == 270:
                       sum10 += magnitude
                   elif nearest45Degrees == 315:
                       sum11 += magnitude

               location = 0

               hn = np.zeros((1, 8))
               h = np.zeros((1, 8))

               h[0][0] = sum4
               h[0][1] = sum5
               h[0][2] = sum6
               h[0][3] = sum7
               h[0][4] = sum8
               h[0][5] = sum9
               h[0][6] = sum10
               h[0][7] = sum11

               for c in range(0, 7):
                   if (h[0][c] > h[0][location]):
                       location = c;

               for i in range(0, 8):
                   hn[0][((6 - 2 + i) % 8)] = h[0][((location + i) % 8)]

               z = [0, 45, 90, 135, 180, 225, 270, 315]
               a = (sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11)
               print()
               print('Normalized Histogram for Corner pixels (part4b): ', (x, y))

               for i in range(0, 8):
                    print(z[i], 'degrees: ', hn[0][i])
               #      plt.title('Normalized Gradient Direction Histogram (part 4b)')
               #      plt.xlabel('Angles (Multiples of 45 degrees)')
               #      plt.ylabel('Magnitudes')
               #      plt.bar(z[i], hn[0][i], width=1, align='center', color='brown')
               # plt.show()


    plt.title("Corner Detection Part 4a")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

my_harris(3, 0.04, 0.30)