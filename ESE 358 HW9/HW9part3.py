import numpy as np
import math

#To find the minimum value from the array with all median values
def minValueArray(array):
    minimum = min(array)
    print('The median-residue using all six trials to find its minimum of all medians is ', minimum)

B = np.zeros((2,2))

j = 1;
listofMedians = []

while j < 7:
    median = 0;
    #To initialize A with these point values again in a loop even if those points chosen were deleted
    #from the previous iterations
    A = np.array([[2, 1], [3, 5], [6, 9], [12, 7], [15, 20], [18, 16], [22, 30]])

    #Responsible for picking two points without picking that same point two times
    randomRows = np.random.choice(7, size=2, replace=False)
    B = A[randomRows, :]

    #Delete the two selected chosen points
    A = np.delete(A, randomRows[0], 0)
    A = np.delete(A, randomRows[1] - 1, 0)

    a_prime = B[0][1] - B[1][1]  #a' = (y2-y1)
    b_prime = B[1][0] - B[0][0]  #b' = (x1-x2)
    c_prime = (B[1][1]*(B[0][0] - B[1][0])) - (B[1][0]*(B[0][1] - B[1][1]))  #c' = y1(x2-x1)-x1(y2-y1)

    a = a_prime / (math.sqrt(math.pow(a_prime, 2) + math.pow(b_prime, 2)))
    b = b_prime / (math.sqrt(math.pow(a_prime, 2) + math.pow(b_prime, 2)))
    c = c_prime / (math.sqrt(math.pow(a_prime, 2) + math.pow(b_prime, 2)))

    point1 = (a * A[0][0]) + (b * A[0][1]) + c
    point2 = (a * A[1][0]) + (b * A[1][1]) + c
    point3 = (a * A[2][0]) + (b * A[2][1]) + c
    point4 = (a * A[3][0]) + (b * A[3][1]) + c
    point5 = (a * A[4][0]) + (b * A[4][1]) + c

    point1 = math.pow(point1, 2)
    point2 = math.pow(point2, 2)
    point3 = math.pow(point3, 2)
    point4 = math.pow(point4, 2)
    point5 = math.pow(point5, 2)

    C = np.array([point1, point2, point3, point4, point5])

    #To sort an array of points to find the median (middle value)
    for i in range(len(C) - 1):
        if(C[i] > C[i+1]):
            temp = C[i]  # temp = 5
            C[i] = C[i+1]  #c[i] = 3
            C[i+1] = temp  # C[i+1] = 5
            median = C[2]

    #Check if the median is less than the threshold: '2'
    if median < 2:
        print('The median residual that satisfies least median squares regression and is less than threshold 2 is', median)
        break;
    #if none of the medians for each trial is less than 2, then find the minimum value of all the medians instead on the
    #sixth trial
    elif j == 6:
        minValueArray(listofMedians)
        break;

    listofMedians.append(median)
    print('CONTINUE with rest out of six trials to try and find median residue that is less than threshold: 2  ')
    j += 1