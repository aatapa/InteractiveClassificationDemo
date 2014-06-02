
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

#img_orig=mpimg.imread('IMG_7459_osakuva.png')
#Xmat = np.loadtxt('features_IMG_7459_osakuva.txt')


img_orig=mpimg.imread('198023.jpg')
Xmat = np.loadtxt('features_198023.txt')

Xmat = np.nan_to_num(Xmat)

#img_orig[:,:,1]=0
#img_orig[:,:,2]=0
#featgen.create_features(img_orig[:,:,0], 9, 1, 1, 8, 1, 1)
#foo
img=img_orig.copy()

pointrange = np.arange(img.shape[0]*img.shape[1])
rows, cols = np.unravel_index(pointrange, (img.shape[0], img.shape[1]))
coords_to_ind = np.vstack([np.hstack([pointrange.reshape(img.shape[0], img.shape[1]), -1 * np.ones((img.shape[0], 1))]), -1 * np.ones((1, img.shape[1]+1))]).astype(int)
pcoll = np.vstack([cols, rows]).T
#Xmat = np.array(img.copy(), dtype = np.int64)
#Xmat = Xmat.reshape((img.shape[0]*img.shape[1],img.shape[2]))
#Xmat = np.hstack([Xmat, pcoll])
classcount = 2

kwargs = {}
bias = 0.

kwargs['train_features'] = Xmat
kwargs['train_labels'] = np.zeros((Xmat.shape[0]), dtype = np.int32)
kwargs['kernel'] = 'GaussianKernel'
kwargs['bias'] = bias
kwargs['gamma'] = 2. ** (-17.)
kwargs['regparam'] = 2. ** (1.)
kwargs['number_of_clusters'] = classcount
kwargs['basis_vectors'] = pyrandom.sample(range(Xmat.shape[0]), 100)
#print kwargs['basis_vectors']

mmc = InteractiveRlsClassifier.createLearner(**kwargs)

plt.ion()

mmc.train()

mmc.working_set = None
mmc.wsc = None


slider_coords = [0.25, 0.06, 0.65, 0.02]
obj_fun_display_coords = [0.25, 0.96, 0.65, 0.02]

class SelectFromCollection(object):
    
    def __init__(self, ax, collection, mmc, img):
        self.imdata = plt.gca().imshow(img)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.yaxis.set_tick_params(size=0)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.xaxis.set_tick_params(size=0)
        self.img = img
        self.canvas = ax.figure.canvas
        self.collection = collection
        #self.alpha_other = alpha_other
        self.mmc = mmc
        self.prevnewclazz = None

        self.xys = collection
        self.Npts = len(self.xys)
        
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
        self.slider_axis = plt.axes(slider_coords, visible = False)
        self.slider_axis2 = plt.axes(obj_fun_display_coords, visible = False)
        self.in_selection_slider = None
        newws = list(set(range(len(self.collection))) - self.lockedset)
        self.mmc.new_working_set(newws)
        self.lasso.line.set_visible(False)
    
    def onselect(self, verts):
        self.path = Path(verts)
        self.ind = np.nonzero(self.path.contains_points(self.xys))[0]
        print 'Selected '+str(len(self.ind))+' points'
        newws = list(set(self.ind) - self.lockedset)
        self.mmc.new_working_set(newws)
        self.redrawall()
    
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
        print 'You pressed', event.key
        if event.key == '1':
            print 'Assigned all selected points to class 1'
            newclazz = 1
            mmc.claim_all_points_in_working_set(newclazz)
        if event.key == '0':
            print 'Assigned all selected points to class 0'
            newclazz = 0
            mmc.claim_all_points_in_working_set(newclazz)
        if event.key == 'a':
            print 'Selected all points'
            newws = list(set(range(len(self.collection))) - self.lockedset)
            self.mmc.new_working_set(newws)
            self.lasso.line.set_visible(False)
        if event.key == 'c':
            changecount = mmc.cyclic_descent_in_working_set()
            print 'Performed ', changecount, 'cyclic descent steps'
        if event.key == 'l':
            print 'Locked the class labels of selected points'
            self.lockedset = self.lockedset | set(self.ind)
            newws = list(set(self.ind) - self.lockedset)
            self.mmc.new_working_set(newws)
            #print newws
        if event.key == 'u':
            print 'Unlocked the selected points'
            self.lockedset = self.lockedset - set(self.ind)
            newws = list(set(self.ind) - self.lockedset)
            self.mmc.new_working_set(newws)
        self.redrawall()
    
    def redrawall(self, newslider = True):
        if newslider:
            nozeros = np.nonzero(self.mmc.classvec_ws)[0]
            self.slider_axis.cla()
            del self.slider_axis
            del self.slider_axis2
            self.slider_axis = plt.axes(slider_coords)
            self.slider_axis2 = plt.axes(obj_fun_display_coords)
            steepness_vector = mmc.compute_steepness_vector()
            X = [steepness_vector, steepness_vector]
            #right = left+width
            #self.slider_axis2.imshow(X, interpolation='bicubic', cmap=plt.get_cmap("Oranges"), alpha=1)
            self.slider_axis2.imshow(X, cmap=plt.get_cmap("Oranges"))
            self.slider_axis2.set_aspect('auto')
            plt.setp(self.slider_axis2.get_yticklabels(), visible=False)
            self.slider_axis2.yaxis.set_tick_params(size=0)
            del self.in_selection_slider
            self.in_selection_slider = None
            self.in_selection_slider = Slider(self.slider_axis, 'Fraction slider', 0., len(mmc.working_set), valinit=len(nozeros))
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
                print 'Claimed', claims, 'points for class', newclazz   #val, nonzeroc, claims
                self.claims = claims
                mmc.claim_n_points(claims, newclazz)
                steepness_vector = mmc.compute_steepness_vector()
                X = [steepness_vector, steepness_vector]
                self.slider_axis2.imshow(X, cmap=plt.get_cmap("Oranges"))
                self.slider_axis2.set_aspect('auto')
                self.redrawall(newslider = False) #HACK!
                self.prevnewclazz = newclazz
            self.in_selection_slider.on_changed(sliderupdate)
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
        self.imdata.set_data(self.img)
        
        if self.lasso.useblit:
            self.lasso.canvas.restore_region(self.lasso.background)
            self.lasso.ax.draw_artist(self.lasso.line)
            self.lasso.canvas.blit(self.lasso.ax.bbox)
        else:
            self.lasso.canvas.draw_idle()
        plt.draw()
        print_instructions()
    
    def disconnect(self):
        self.lasso.disconnect_events()
        self.canvas.draw_idle()



def print_instructions():
    print
    print 'Draw a selection by holding down the left mouse button'
    print 'Press the Fraction slider with the left mouse button to claim points inside the selection'
    print 'Press a to select all points in the figure'
    print 'Press 1 to claim all points in selection into class 1'
    print 'Press 0 to claim all points in selection into class 0'
    print 'Press l to lock all selected points to their current classes (e.g. they can not be claimed)'
    print 'Press u to unlock all selected points after which they can be claimed again'
    print 'Press c to perform a cyclic descent in the selection'
    print


selector = SelectFromCollection(plt.gca(), pcoll, mmc, img)

plt.draw()
       
selector.redrawall()
print 'All points are selected'

plt.show(block=True)

