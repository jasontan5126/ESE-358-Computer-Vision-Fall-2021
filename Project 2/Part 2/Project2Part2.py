"""
Jason Tan
112319102

ESE 358 Project 2
Python Template
Stable version: python 3.8

Don't touch the import statements. The skeleton code uses numpy and cv2.

Most IDE's will ask to install these packages and will just work. Here are some links in case they don't
and you need to install them manually. You might need to refresh your IDE after installing if you're still seeing errors

cv2 package: https://pypi.org/project/opencv-python/
    For python 3 just run 'pip3 install opencv-python' from your terminal to install the package
    numpy package: https://numpy.org/install/
    When you install cv2, it should come with all of the numpy libraries
"""
import math
import sys
import numpy
import numpy as np
import cv2


# Predefined functions called below. There are some pieces that you are required to fill here too

'''
function for rotation and translation
'''
def Map2Da(K, R, T, Vi):
    T_transpose = np.transpose(np.atleast_2d(T)) #numpy needs to treat 1D as 2D to transpose
    V_transpose = np.transpose(np.atleast_2d(np.append(Vi,[1])))
    RandTappended = np.append(R, T_transpose, axis=1)
    P_1 = np.dot(K, RandTappended)
    P = np.dot(P_1, V_transpose)  #K * RandTappended * V_transpose
    P = np.asarray(P).flatten() #Turn back into 1D array
    #Equivalent matlab code for above -->  P=K*[R T']*[Vi 1]';
    #Some projection matrix formula

    w1 = P[2]
    v= [None]*2 #makes an empty array of size 2

    v[0]= P[0] / w1
    v[1] = P[1] / w1
    # v[1] = ????????????? - Solution here

    return v


'''
function for mapping image coordinates in mm to
row and column index of the image, with pixel size p mm and
image center at [r0,c0]
'''
def MapIndex(u, c0, r0, p):
    v = [None]*2
    v[0] = round(r0 - u[1] / p)
    v[1] = round(c0 + u[0] / p)
    return v


'''
Wrapper for drawing line cv2 draw line function
Necessary to flip the coordinates b/c of how Python indexes pixels on the screen >:(
A : matrix to draw a line in
vertex1 : 1D array in (x,y) format to be displayed on screen
vertex2 : 1D array in (x,y) format to be displayed on screen
thickness : thickness of the line(default = 5)
color : RGB tuple for the line to be drawn in (default = (255, 0, 0 ) ie white)

@return : the matrix with the line drawn in it

NOTE: order of vertex1 and vertex2 does not change the line drawn
'''

def drawLine(A,vertex1, vertex2, color = (255, 0, 0), thickness=3):
    v1 = list(reversed(vertex1))
    v2 = list(reversed(vertex2))
    return cv2.line(A, v1, v2,  color, thickness)



# define 8 points of the cube in world coordinate
def main():
    length = 10
    V1 = np.array([0, 0, 0])
    V2 = np.array([0, length, 0])
    V3 = np.array([length, length, 0])
    V4 = np.array([length, 0, 0])
    V5 = np.array([length, 0, -length])
    V6 = np.array([0, length, -length])
    V7 = np.array([0, 0, -length])
    V8 = np.array([length, length, -length])

    '''
    Find the unit vector u81 corresponding to the axis of rotation which is along (V8-V1).
    From u81, compute the 3x3 matrix N in Eq. 2.32 used for computing the rotation matrix R in eq. 2.34
    '''

    u81 = (V8 - V1)/(math.sqrt(pow((V8[0] - V1[0]), 2) + pow((V8[1] - V1[1]), 2) + pow((V8[2] - V1[2]), 2)))

    N = [[0, -u81[2], u81[1]],
         [u81[2], 0, -u81[0]],
         [-u81[1], u81[0], 0]]

    N_int = np.array(N, dtype='int')


    # ????????????????? - Solution here

    T0 = np.array([-20, -25, 500])  # origin of object coordinate system in mm
    # T0 = [-30, -20, 500] #origin of object coordinate system

    # set given values

    #f = 40  # focal length in mm

    #Have the user enter any focal length value in mm
    f = int(input('Enter f: '))

    # Initialize the 3x3 camera matrix K given the focal length

    # K = ???????????????? - Solution here
    K = [[f, 0, 0], [0, f, 0], [0, 0, 1]]


    velocity = np.array([2, 9, 7])  # translational velocity
    theta0 = 0

    w0 = 20  # angular velocity in deg/sec

    p = 0.01  # pixel size(mm)
    Rows = 600  # image size
    Cols = 600  # image size
    A = np.zeros((Rows, Cols), dtype=np.uint8) #output image
    r0 = np.round(Rows / 2)
    c0 = np.round(Cols / 2)

    '''
     You are given a rectangle/square in 3D space specified by its
     corners at 3D position vectors V1, V2, V3, V4.
     You are also given a rectangular/square graylevel image
     tmap of size r x c.
     This image is to be "painted" on the 3D rectangle/square, and
     for each pixel at position (i,j),
     the corresponding 3D coordinates
     X(i,j), Y(i,j), and Z(i,j), should be computed,
     and that 3D point is
     associated with the brightness given by tmap(i,j).

     Find the unit vectors corresponding to u21=(V2-V1)/|(V2-V1)|
     and u41= (V4-V1)/|(v4-V1), and compute X(i,j), Y(i,j), and Z(i,j).
     Compute the unit vector u21 along (V2-V1) and
     Compute the unit vector u41 along (V4-V1) and
    '''

    # h=?  height = distance from v2 to v1
    # w=?  width = distance from v4 to v1
    h = math.sqrt(pow((V2[0] - V1[0]), 2) + pow((V2[1] - V1[1]), 2) + pow((V2[2] - V1[2]), 2))
    w = math.sqrt(pow((V4[0] - V1[0]), 2) + pow((V4[1] - V1[1]), 2) + pow((V4[2] - V1[2]), 2))
    # print(w)
    u21 = (V2-V1)/h
    u41 = (V4-V1)/w
    # u21 =??????????????? - Solution here
    # u41 =??????????????? - Solution here

    # For each pixel of texture map, compute its (X,Y,Z) values
    background = cv2.imread('project2Part2.jpg')  # texture map image
    if background is None:
        print("image file can not be found on path given. Exiting now")
        sys.exit(1)


    # For each pixel of texture map, compute its (X,Y,Z) values
    tmap = cv2.imread('einstein50x50v.jpg')  # texture map image
    if tmap is None:
        print("image file can not be found on path given. Exiting now")
        sys.exit(1)

    r, c, colors = tmap.shape
    X = np.zeros((r, c), dtype=np.float64)
    Y = np.zeros((r, c), dtype=np.float64)
    Z = np.zeros((r, c), dtype=np.float64)
    for i in range(0, r):
        for j in range(0, c):
            p1 = V1 + (i) * u21 * (h / r) + (j) * u41 * (w / c)
            X[i, j] = p1[0]
            Y[i, j] = p1[1]
            Z[i, j] = p1[2]
            # Y[i,j] = ?? - Solution here
            # Z[i,j] = ?? - Solution here

    acc = np.array([0.0, -0.80, 0])  # acceleration
    time_range = np.arange(0.0, 24.2, 0.2)
    for t in time_range:  # Generate a sequence of images as a function of time
        theta = theta0 + w0 * t
        T = T0 + velocity * t + 0.5 * acc * t * t
        # Compute the Rotation matrix from N and theta (Be mindful of radians vs degrees!)

        # R = ????????????? - Solution here
        #N0 is passing through V1 and V8, and pointing towards V8 (i.e. , N0 is parallel to V8-V1).
        #N_x = [1,1, -1]
        #Rotation matrix formula: 2.34 from textbook

        identityMatrix = (numpy.identity(3, dtype=float))
        radians = numpy.deg2rad(theta)
        eq = (math.sin(radians)*numpy.dot(N, 1)) + ((1 - math.cos(radians))*numpy.dot(N,N))
        R = identityMatrix + eq    #rotation matrix

        # find the image position of vertices
        v = Map2Da(K, R, T, V1)
        v1 = MapIndex(v, c0, r0, p)
        v = Map2Da(K, R, T, V2)
        v2 = MapIndex(v, c0, r0, p)
        v = Map2Da(K, R, T, V3)
        v3 = MapIndex(v, c0, r0, p)
        v = Map2Da(K, R, T, V4)
        v4 = MapIndex(v, c0, r0, p)
        v = Map2Da(K, R, T, V5)
        v5 = MapIndex(v, c0, r0, p)
        v = Map2Da(K, R, T, V6)
        v6 = MapIndex(v, c0, r0, p)
        v = Map2Da(K, R, T, V7)
        v7 = MapIndex(v, c0, r0, p)
        v = Map2Da(K, R, T, V8)
        v8 = MapIndex(v, c0, r0, p)

        # ????????????????????????????? - Solution here

        # Draw edges of the cube
        color = (255, 0, 0)
        thickness = 3

        A = np.zeros((Rows, Cols), dtype=np.uint8)

        # ???????????????????????????? - Solution here to get background image
        for ro in range(600):
            for col in range(600):
                A[ro, col] = background[ro, col, 0]


        #Draw the edges that connect each vertices to make a 3D cube
        A = drawLine(A, v1, v2, color, thickness)
        A = drawLine(A, v2, v3, color, thickness)
        A = drawLine(A, v3, v4, color, thickness)
        A = drawLine(A, v4, v1, color, thickness)
        A = drawLine(A, v7, v6, color, thickness)
        A = drawLine(A, v6, v8, color, thickness)
        A = drawLine(A, v8, v5, color, thickness)
        A = drawLine(A, v5, v7, color, thickness)
        A = drawLine(A, v1, v7, color, thickness)
        A = drawLine(A, v2, v6, color, thickness)
        A = drawLine(A, v3, v8, color, thickness)
        A = drawLine(A, v4, v5, color, thickness)
        # ???????????????????????????? - Solution here

        # Add texture map to one face.
        for i in range(r):
            for j in range(c):
                p1 = [X[i, j], Y[i, j], Z[i, j]]
                v = Map2Da(K, R, T, p1)
                index = MapIndex(v, c0, r0, p)
                ir = index[0]
                jr = index[1]

                # Find the row and column indices [ir,jr] in integers that give #the image position of point p1 in A.
                # Use the same method as for the corners of the cube above.

                # ????????????????????????? - Solution here

                if ((ir >= 0) and (jr >= 0) and (ir < Rows) and (jr < Cols)):
                    A[ir ,jr] = tmap[i, j, 0]
                # In a general case, you may need to
                # fill up gaps in A[ir,jr]
                # through interpolation. But, in this project,
                # you can skip interpolation. The output will not
                # look nice due to the gaps.


        A = cv2.cvtColor(A, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Display Window", A)

        #cv2.waitKey(0)
        # ^^^ uncomment if you want to display frame by frame
        # and press return(or any other key) to display the next frame
        #by default just waits 1 ms and goes to next frame
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
