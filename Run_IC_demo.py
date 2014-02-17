
import cProfile
import random as pyrandom
import sys
import time

from numpy import meshgrid,reshape,linspace,ones,min,max,concatenate,transpose,mat,float64,zeros, array, multiply
import numpy as np

import matplotlib
matplotlib.rcParams['keymap.yscale'] = ''
import matplotlib.pylab as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import figure, show, cm
from matplotlib.widgets import LassoSelector, Slider
from matplotlib.path import Path
from scipy import sparse

from rlscore.learner.interactive_rls_classifier import InteractiveRlsClassifier

#import featgen

#img_orig=mpimg.imread('8068.jpg')
#img_orig=mpimg.imread('299091.jpg')

img_orig=mpimg.imread('198023.jpg')
Xmat = np.loadtxt('features_198023.txt')
Xmat = np.nan_to_num(Xmat)

#img_orig[:,:,1]=0
#img_orig[:,:,2]=0
#featgen.create_features(img_orig[:,:,0], 9, 1, 1, 8, 1, 1)
#foo
img=img_orig.copy()
#print img.shape, img.dtype
#print Xmat
#Xmat = Xmat[:, 2:]
#print Xmat
#print Xmat.shape

pointrange = np.arange(img.shape[0]*img.shape[1])
rows, cols = np.unravel_index(pointrange, (img.shape[0], img.shape[1]))
coords_to_ind = np.vstack([np.hstack([pointrange.reshape(img.shape[0], img.shape[1]), -1 * np.ones((img.shape[0], 1))]), -1 * np.ones((1, img.shape[1]+1))]).astype(int)
pcoll = np.vstack([cols, rows]).T
#Xmat = np.array(img.copy(), dtype = np.int64)
#Xmat = Xmat.reshape((img.shape[0]*img.shape[1],img.shape[2]))
#Xmat = np.hstack([Xmat, pcoll])
classcount = 2

rpool = {}
bias = 0.

rpool['train_features'] = Xmat
rpool['train_labels'] = np.zeros((Xmat.shape[0]), dtype = np.int32)
rpool['kernel'] = 'GaussianKernel'
rpool['bias'] = bias
rpool['gamma'] = 2. ** (-17.)
rpool['regparam'] = 2. ** (1.)
rpool['number_of_clusters'] = classcount
rpool['basis_vectors'] = pyrandom.sample(range(Xmat.shape[0]), 100)
#print rpool['basis_vectors']

mmc = InteractiveRlsClassifier.createLearner(**rpool)

plt.ion()
#plt.plot([1,2],[3,4],'r^')

mmc.train()
#print mmc.resource_pool

mmc.working_set = None
mmc.wsc = None

#axcolor = 'lightgoldenrodyellow'


class SelectFromCollection(object):
    
    def __init__(self, ax, collection, mmc, img):
        self.imdata = plt.gca().imshow(img)
        self.img = img
        self.canvas = ax.figure.canvas
        self.collection = collection
        #self.alpha_other = alpha_other
        self.mmc = mmc
        self.prevnewclazz = None

        self.xys = collection
        self.Npts = len(self.xys)
        
        self.neighset = set([])
        self.lockedset = set([])

        self.lasso = LassoSelector(ax, onselect=self.onselect)#, lineprops = {:'prism'})
        self.lasso.disconnect_events()
        self.lasso.connect_event('button_press_event', self.lasso.onpress)
        self.lasso.connect_event('button_release_event', self.onrelease)
        self.lasso.connect_event('motion_notify_event', self.lasso.onmove)
        self.lasso.connect_event('draw_event', self.lasso.update_background)
        self.lasso.connect_event('key_press_event', self.onkeypressed)
        #self.lasso.connect_event('button_release_event', self.onrelease)
        self.ind = []
        self.slider_axis = plt.axes([0.25, 0.1, 0.65, 0.03], visible = False)
        self.slider_axis2 = plt.axes([0.25, 0.07, 0.65, 0.03], visible = False)
        #self.in_selection_slider = Slider(self.slider_axis, 'Fraction', 0., 30.0, valinit=f0)
        #    #fig.canvas.draw_idle()
        #self.in_selection_slider.on_changed(sliderupdate)
        #self.slider_axis = None
        self.in_selection_slider = None
    
    def newSlider(self):
        #print 'newslider'
        nozeros = np.nonzero(self.mmc.classvec_ws)[0]
        self.slider_axis.cla()
        del self.slider_axis
        del self.slider_axis2
        self.slider_axis = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.slider_axis2 = plt.axes([0.25, 0.07, 0.65, 0.03])
        steepness_vector = mmc.compute_steepness_vector()
        X = [steepness_vector, steepness_vector]
        #right = left+width
        #self.slider_axis2.imshow(X, interpolation='bicubic', cmap=plt.get_cmap("Blues"), alpha=1)
        self.slider_axis2.imshow(X, cmap=plt.get_cmap("Blues"))
        self.slider_axis2.set_aspect('auto')
        del self.in_selection_slider
        self.in_selection_slider = None
        self.in_selection_slider = Slider(self.slider_axis, 'Fraction', 0., len(mmc.working_set), valinit=len(nozeros))
        def sliderupdate(val):
            val = int(val)
            nonzeroc = len(np.nonzero(self.mmc.classvec_ws)[0])
            if val > nonzeroc:
                claims = val - nonzeroc
                newclazz = 1
            elif val < nonzeroc:
                claims = nonzeroc - val
                newclazz = 0
            else: return
            print val, nonzeroc, claims
            self.claims = claims
            mmc.claim_n_points(claims, newclazz)
            steepness_vector = mmc.compute_steepness_vector()
            X = [steepness_vector, steepness_vector]
            self.slider_axis2.imshow(X, cmap=plt.get_cmap("Blues"))
            self.slider_axis2.set_aspect('auto')
            self.recolor()
            self.prevnewclazz = newclazz
        self.in_selection_slider.on_changed(sliderupdate)
        self.redrawline()
    
    def onselect(self, verts):
        self.path = Path(verts)
        self.ind = np.nonzero(self.path.contains_points(self.xys))[0]
        print 'Selected '+str(len(self.ind))+' points'
        newws = list(set(self.ind) - self.lockedset)
        self.mmc.new_working_set(newws)
        self.newSlider()
    
    def onpress(self, event):
        if self.lasso.ignore(event) or event.inaxes != self.ax:
            return
        self.lasso.line.set_data([[], []])
        self.lasso.verts = [(event.xdata, event.ydata)]
        self.lasso.line.set_visible(True)

    def onrelease(self, event):
        if self.lasso.ignore(event):
            return
        if self.lasso.verts is not None:
            if event.inaxes == self.lasso.ax:
                self.lasso.verts.append((event.xdata, event.ydata))
            self.lasso.onselect(self.lasso.verts)
        self.lasso.verts = None
    
    def onkeypressed(self, event):
        print('you pressed', event.key) #, event.xdata, event.ydata
        if event.key.isdigit() and mmc.working_set != None:
            newclazz = int(event.key)
            if newclazz >= 0 and newclazz < classcount:
                
                #self.mmc.new_working_set(list(self.neighset))
                
                steepestdir = mmc.claim_a_point(newclazz)
                col, row = self.collection[steepestdir]
                print steepestdir, row, col
                self.recolor()
                
                '''
                self.neighset.discard(coords_to_ind[row, col])
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        nind = coords_to_ind[row + i, col + j]
                        if mmc.classvec[nind] == 0: self.neighset.add(nind)
                print self.neighset
                '''
                
                self.prevnewclazz = newclazz
        if event.key == 'a': # and self.prevnewclazz != None:
            newclazz = 1
            mmc.claim_all_points_in_working_set(newclazz)
            #col_row = self.collection[mmc.working_set]
            self.recolor()
            #nonzeros_ws = np.nonzero(self.mmc.classvec_ws)[0]
            '''
            self.neighset -= set(self.mmc.classvec_ws)
            img.shape[0], img.shape[1]
            for pind in range(col_row.shape[0]):
                colind, rowind = col_row[pind, 1], col_row[pind, 0]
                #if rowind > 0:
                #    if colind > 0:
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        nind = coords_to_ind[colind + j, rowind + i]
                        if mmc.classvec[nind] == 0: self.neighset.add(nind)
            print self.neighset'''
        if event.key == 'n': # and self.prevnewclazz != None:
            newclazz = 0
            mmc.claim_all_points_in_working_set(newclazz)
            #col_row = self.collection[mmc.working_set]
            self.recolor()
        if event.key == 'c':
            changecount = mmc.cyclic_descent_in_working_set()
            print changecount
            self.recolor()
        if event.key == 'l':
            self.lockedset = self.lockedset | set(self.ind)
            newws = list(set(self.ind) - self.lockedset)
            self.mmc.new_working_set(newws)
            print newws
        if event.key == 'u':
            self.lockedset = self.lockedset - set(self.ind)
            newws = list(set(self.ind) - self.lockedset)
            self.mmc.new_working_set(newws)
        self.newSlider()
    
    def recolor(self):
        oneclazz = np.nonzero(self.mmc.classvec)[0]
        col_row = self.collection[oneclazz]
        rowcs, colcs = col_row[:, 1], col_row[:, 0]
        #self.img[rowcs, colcs, :] = img_orig[rowcs, colcs, :] + 128
        #self.img[rowcs, colcs, :] = 128
        self.img[rowcs, colcs, :] = 0
        self.img[rowcs, colcs, 0] = 255
        zeroclazz = np.nonzero(self.mmc.classvec - 1)[0]
        col_row = self.collection[zeroclazz]
        rowcs, colcs = col_row[:, 1], col_row[:, 0]
        self.img[rowcs, colcs, :] = img_orig[rowcs, colcs, :]
        '''
        if newclazz == 1:
            #self.img[rowcs, colcs, :] = 0
            #print self.img[rowcs, colcs, 0]
            #self.img[rowcs, colcs, 0] = np.fmin(255, self.img[rowcs, colcs, 0]+50)
            self.img[rowcs, colcs, :] += 128
            #print self.img[rowcs, colcs, 0]
        else:
            self.img[rowcs, colcs, :] = img_orig[rowcs, colcs, :]
        '''
        self.imdata.set_data(self.img)
        self.redrawline()
    
    def disconnect(self):
        self.lasso.disconnect_events()
        self.canvas.draw_idle()
    
    def redrawline(self):
        if self.lasso.useblit:
            self.lasso.canvas.restore_region(self.lasso.background)
            self.lasso.ax.draw_artist(self.lasso.line)
            self.lasso.canvas.blit(self.lasso.ax.bbox)
        else:
            self.lasso.canvas.draw_idle()
        plt.draw()


def ravel_index(pos, shape):
    res = 0
    acc = 1
    for pi, si in zip(reversed(pos), reversed(shape)):
        res += pi * acc
        acc *= si
    return res



selector = SelectFromCollection(plt.gca(), pcoll, mmc, img)
'''def foo():
            rows, cols = [], []
            for i in range(selector.claims):
                steepestdir = mmc.claim_a_point(1)
                col, row = selector.collection[steepestdir]
                #print steepestdir, row, col
                rows.append(row)
                cols.append(col)
            selector.recolor()'''

plt.draw()

plt.show(block=True)

