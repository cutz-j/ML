### Assignment 1 ###

import numpy as np
import matplotlib.pyplot as plt

## Q1: eye matrix ##
def eye(num):
    resList = []
    for i in range(num):
        tempList = []
        for j in range(num):
            if i == j:
                tempList.append(1)
            else:
                tempList.append(0)
        resList.append(tempList)
    return np.array(resList)
print(eye(5))

## Q2: Linear regression with one variable ##
