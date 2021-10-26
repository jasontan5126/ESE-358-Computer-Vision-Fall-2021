"""
ESE 358 Homework 9 Linear Least Squares
"""

import numpy as np

x = np.array([2,3,6,12,15,18,22])
y = np.array([1,5,9,7,20,16, 30])


A = np.vstack([x,np.ones(len(x))]).T

p = np.linalg.matrix_power((np.dot(np.transpose(A), A)), -1)
p = np.dot(p, np.dot(np.transpose(A), y))
print('[m, b] = ', p)