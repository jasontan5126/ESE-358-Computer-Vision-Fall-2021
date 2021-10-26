#ESE 358 Homework 9 Total Linear Least Squares Part 2


import numpy as np
from numpy.linalg import eig

xbar= (2+3+6+12+15+18+22)/7
ybar= (1+5+9+7+20+16+30)/7
A= np.vstack([[2 - xbar, 3 - xbar, 6 - xbar,12-xbar,15-xbar,18-xbar,22-xbar],[1-ybar,5-ybar,9-ybar,7-ybar,20-ybar,16-ybar, 30-ybar]]).T

B = np.dot(np.transpose(A), A)
w,v=eig(B)

#print('Eignenvalues: ', w)
#print('Eigenvectors: ', v)

if w[0] < w[1]:
    x = v[:, 0]
    c = (-v[0][0]*xbar)-(v[0][1]*ybar)
    print('a = ',v[0][0], 'b = ', v[0][1], 'c = ', c)
elif w[0] > w[1]:
    x = v[:, 1]
    c = (-v[1][0] * xbar) - (v[1][1] * ybar)
    print('a = ', v[1][0], 'b = ', v[1][1], 'c = ', c)