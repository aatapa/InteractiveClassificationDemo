import sys
import random
import numpy as numpy
import math

random.seed(200)

class GaussianMoreClusters2D():
    def __init__(self):
        pass

    def generate(self,num,mux,muy,var):
        Y = []
        X = []
    
        for j in range(len(num)):
            for i in range(num[j]):
                # Generate "random" order of samples
                x = random.gauss(mux[j],var[j])
                y = random.gauss(muy[j],var[j])
                X.append([x,y])
                Y.append(j)
        
        return Y, X
#        X_shuffled, Y_shuffled = [], []
#        indices = range(len(X))
#        random.shuffle(indices)
#        for i in range(len(X)):
#            X_shuffled.append(X[indices[i]])
#            Y_shuffled.append(Y[indices[i]])
#        
#        return Y_shuffled, X_shuffled
