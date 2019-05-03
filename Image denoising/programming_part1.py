# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:42:21 2019

@author: liuyu
"""

from skimage.io import imread, imsave
import numpy as np
from sklearn.metrics import confusion_matrix

# transfer 0, 255 to -1,1 or reversely convert {-1, 1} matrix to image
def tranValues(x_input, value_dict):
    x = []
    for key in np.nditer(x_input):
        x.append(value_dict[int(key)])
    x = np.reshape(x, x_input.shape)
    return x

# get the neighbour's value of one pixel 
def getNeighbors_(x, rowid, colid):
    # the limit of the row col id for the neib
    rowMin, rowMax = 0, x.shape[0] - 1
    colMin, colMax = 0, x.shape[1] - 1
    
    # upNei, downNei, leftNei, rightNei
    ret = {"upNei": [rowid - 1, colid], "downNei": [rowid + 1, colid], "leftNei": [rowid, colid - 1], "rightNei": [rowid, colid + 1]}

    for item in ret.items():
        k, v, flag = item[0], item[1], True
        if v[0] > rowMax or v[0] < rowMin: flag = False
        if v[1] > colMax or v[1] < colMin: flag = False
        
        if flag: 
            ret[k] = x[v[0],v[1]]
        else:
            ret[k] = 0
    return ret

# compute one pixel's energy according to the energy function
def get1Energy_(xi, yi, neighbors, h, Beta, Eta):
    product1 = h * xi
    
    product2 = Beta * xi * (neighbors["upNei"] + neighbors["downNei"] + neighbors["leftNei"] + neighbors["rightNei"])
    
    product3 = Eta * (yi * xi)
    ret = product1 - product2 - product3
    return ret

# compute all pixels energy
def compEnergy_(x, y, h, Beta, Eta):
    nrow, ncol = x.shape
    ret = 0
    for rowid in range(nrow):
        for colid in range(ncol):
            xi, yi = x[rowid, colid], y[rowid, colid]
            neighbors = getNeighbors_(x, rowid, colid)
            ret += get1Energy_(xi, yi, neighbors, h, Beta, Eta)
    return ret

def deNoise(x_input, maxIter, h, Beta, Eta):
    # copy the figure and assign memory for the previous energy and new energy after flipped pixels
    x, y, x_pre = x_input.copy(), x_input.copy(), x_input.copy()
    oldSumEnergy, newSumEnergy = compEnergy_(x, y, h, Beta, Eta), 0
    
    # times run the algorithm
    for iterTime in range(maxIter):            
        nrow, ncol = x.shape
        
        # iter every pixel in the figure
        for rowid in range(nrow):
            for colid in range(ncol):
                xi, yi = x[rowid, colid], y[rowid, colid]
                neighbors = getNeighbors_(x, rowid, colid)
                oldEnergy = get1Energy_(xi, yi, neighbors, h, Beta, Eta)
                #print(oldEnergy)
                newEnergy = get1Energy_(-xi, yi, neighbors, h, Beta, Eta)
                #print(newEnergy)
                deltaEnergy = oldEnergy - newEnergy
                #print(deltaEnergy)
                
                if deltaEnergy < 0:
                    x[rowid, colid] = xi
                else:
                    x[rowid, colid] = -xi
        
        # update the new energy after flipped pixels
        newSumEnergy = compEnergy_(x, y, h, Beta, Eta)
        print("iter: ", iterTime, ", previous energy: ", oldSumEnergy, ", new energy: ", newSumEnergy)
        # if new energy is small, update the previous energy
        if newSumEnergy < oldSumEnergy: 
            oldSumEnergy = newSumEnergy
            x_pre = x.copy()
        # if the new evergy is bigger or equal to the previous energy, return the picture to result
        else:
            return x_pre
        
    return x

work_dir = "C:/Users/liuyu/Desktop/Spring2019/CS542/2019spring/problem _set/PS5/Bayes_Lena_Images/Bayes_Lena_Images/"

nosBayes = imread(work_dir + "Bayesnoise_grayscale.png")
oriBayes = imread(work_dir + "Bayes_grayscale.png")

imgToVal = {0: -1, 255: 1}
valToImg = {-1: 0, 1: 255}

x = tranValues(nosBayes, imgToVal)
xc = deNoise(x, 10, .01, 5, 1)
xc_img = tranValues(xc, valToImg)

cf = confusion_matrix(oriBayes.flatten(), xc_img.flatten())
acc = np.sum(np.diag(cf))/np.sum(cf)

print("The confusion matrix is:")
print(cf)
print("The accuracy is:")
print(acc)

imsave("C:/Users/liuyu/Desktop/Spring2019/CS542/2019spring/problem _set/PS5/part1/denoisedBayes__.png", xc_img)