import numpy as np
import random

class Moons:
    # Default parameters
    parameters = {
    'num': 100,
    'ratio': 0.5,
    'center1': [0, 0],
    'center2': [1, 0.5],
    'radius1': 1.0,
    'radius2': 1.0,
    'noise': 0.05,
    'filename': None,
    'seed':0
    }
        
    def __init__(self,  ** kw):
        self.setParameters( ** kw)
        random.seed(self.__seed)

    def setParameters(self,  ** kw):
        for attr, val in kw.items():
            self.parameters[attr] = val
        self.__num = int(self.parameters['num'])
        assert self.__num > 0
        self.__ratio = float(self.parameters['ratio'])
        assert (self.__ratio > 0) & (self.__ratio < 1) 
        self.__center1 = self.parameters['center1']
        self.__center2 = self.parameters['center2']
        self.__radius1 = float(self.parameters['radius1'])
        assert self.__radius1 >= 0
        self.__radius2 = float(self.parameters['radius2'])
        assert self.__radius2 >= 0        
        self.__noise = float(self.parameters['noise'])
        assert self.__noise >= 0
        self.__filename = self.parameters['filename']
        self.__seed = int(self.parameters['seed'])
        
    def generate(self):
        X = []
        Y = []
        num_pos = int(self.__num*self.__ratio)
        num_neg = int(self.__num*(1-self.__ratio))
        if num_pos + num_neg < self.__num:
            num_neg += 1

        for i in xrange(num_pos):
            theta = random.random() * np.pi
            x = self.__center1[0] + self.__radius1 * np.cos(theta)
            y = self.__center1[1] + self.__radius1 * np.sin(theta)
            x = x + random.gauss(0, self.__noise)
            y = y + random.gauss(0, self.__noise)
            X.append([x, y])
            Y.append(1)
        for i in xrange(num_neg):
            theta = random.random() * np.pi
            x = self.__center2[0] + self.__radius2 * np.cos(-theta)
            y = self.__center2[1] + self.__radius2 * np.sin(-theta)
            x = x + random.gauss(0, self.__noise)
            y = y + random.gauss(0, self.__noise)
            X.append([x, y])
            Y.append(-1)   
        
        if self.__filename != None:
            ofile = open(self.__filename, 'w')
            for i in xrange(len(L)):
                ofile.write('%g ' % L[i])
                for j in xrange(len(X[i])):
                    ofile.write(str(j + 1) + ':%+12.5e ' % X[i][j])
                ofile.write('\n')
            ofile.close()                        

        return Y,X

def moons(n=10, seed=0):
    ratio = 0.5
    gen_train = Moons(num=n, ratio=ratio, center1=[0,0], center2=[1,0.1], radius1=1.0, radius2=1.0, noise=0.15, seed=seed)
    X, L = gen_train.generate()
    return X, L

