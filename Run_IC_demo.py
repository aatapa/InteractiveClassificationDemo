
import cProfile
import random as pyrandom
pyrandom.seed(100)
import sys

import numpy as np

import matplotlib
matplotlib.rcParams['keymap.yscale'] = ''
import matplotlib.pylab as plt
import matplotlib.image as mpimg
from matplotlib.widgets import LassoSelector, Slider
from matplotlib.path import Path

from rlscore.learner.interactive_rls_classifier import InteractiveRlsClassifier


def create_grid(num_of_rows, num_of_cols, ws):
    
    """Convenience function for creating a grid of points for an image of size num_of_rows * num_of_cols.

    Parameters
    ----------
    num_of_rows : int
        Number of rows in the original image
        
    num_of_cols : int
        Number of columns in the original image
        
    ws : int
        Determines the size of a window around grid points (2 * windowsize + 1) and accordingly the size of the grid
    
    Returns
    -------
    pcoll : numpy.array
        array containing the (x,y) coordinates of the grid points
    
    incinds : list
        list containing the indices of the grid points relative to the indices of all points.
    """
    pointrange = np.arange(num_of_rows * num_of_cols)
    rg = int(num_of_rows / (2 * ws + 1))
    cg = int(num_of_cols / (2 * ws + 1))
    gridpointrange = np.arange(rg * cg)
    rows, cols = np.unravel_index(gridpointrange, (rg, cg))
    rows, cols = rows * (2 * ws + 1) + ws, cols * (2 * ws + 1) + ws
    pcoll = np.vstack([cols, rows]).T
    incinds = pointrange.reshape((num_of_rows, num_of_cols))[rows, cols]
    return pcoll, incinds

#Load image file and previously created features
img = mpimg.imread('198023.jpg')

windowsize = 5 #Set this value to 0 in order to include all pixels
#Generate pcoll, an array consisting of the (x,y) coords of all points in the image
pcoll, incinds = create_grid(img.shape[0], img.shape[1], windowsize) 

#featgen.create_features(img[:,:,0], 9, 1, 1, 8, 1, 1)
Xmat = np.loadtxt('features_198023.txt')
Xmat = np.nan_to_num(Xmat)

#Comment this if the feature file consists of grid points features only instead of all points
Xmat = Xmat[incinds]

#Ensure that the image has as many points as the feature file
assert pcoll.shape[0] == Xmat.shape[0]

#Uncomment this if coordinates are used as features. STRONG EFFECT!
#Xmat = np.hstack([Xmat, pcoll])

#Create an interactive classifier object
kwargs = {}
kwargs['X'] = Xmat
kwargs['Y'] = np.zeros((Xmat.shape[0]), dtype = np.int32)
kwargs['kernel'] = 'GaussianKernel'
kwargs['bias'] = 0.
kwargs['gamma'] = 2. ** (-17.)
kwargs['regparam'] = 2. ** (1.)
kwargs['number_of_clusters'] = 2
kwargs['basis_vectors'] = Xmat[pyrandom.sample(range(Xmat.shape[0]), 100)]
mmc = InteractiveRlsClassifier(**kwargs)




class SelectFromCollection(object):
    
    """Interactive RLS classifier interface for image segmentation

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The Figure object on which the interface is drawn.
        
    mmc : rlscore.learner.interactive_rls_classifier.InteractiveRlsClassifier
        Interactive RLS classifier object
        
    img : numpy.array
        Array consisting of image data
        
    collection : numpy.array, shape = [n_pixels, 2]
        array consisting of the (x,y) coordinates of all usable pixels in the image
    
    windowsize : int
        Determines the size of a window around grid points (2 * windowsize + 1) 
    """
    
    def __init__(self, fig, mmc, img, collection, windowsize = 0):
        
        #Initialize the main axis
        ax = fig.add_axes([0.1,0.1,0.8,0.8])
        ax.set_yticklabels([])
        ax.yaxis.set_tick_params(size = 0)
        ax.set_xticklabels([])
        ax.xaxis.set_tick_params(size = 0)
        self.imdata = ax.imshow(img)
        
        #Initialize LassoSelector on the main axis
        self.lasso = LassoSelector(ax, onselect = self.onselect)
        self.lasso.connect_event('key_press_event', self.onkeypressed)
        self.lasso.line.set_visible(False)
        
        self.mmc = mmc
        self.img = img
        self.img_orig = img.copy()
        self.collection = collection
        self.selectedset = set([])
        self.lockedset = set([])
        self.windowsize = windowsize
        
        #Initialize the fraction slider
        self.slider_axis = fig.add_axes([0.2, 0.06, 0.6, 0.02])
        self.in_selection_slider = Slider(self.slider_axis,
                                          'Fraction slider',
                                          0.,
                                          1,
                                          valinit = len(np.nonzero(self.mmc.classvec_ws)[0]) / len(mmc.working_set))
        def sliderupdate(val):
            val = int(val * len(mmc.working_set))
            nonzeroc = len(np.nonzero(self.mmc.classvec_ws)[0])
            if val > nonzeroc:
                claims = val - nonzeroc
                newclazz = 1
            elif val < nonzeroc:
                claims = nonzeroc - val
                newclazz = 0
            else: return
            print('Claimed', claims, 'points for class', newclazz)
            self.claims = claims
            mmc.claim_n_points(claims, newclazz)
            self.redrawall()
        self.in_selection_slider.on_changed(sliderupdate)
        
        #Initialize the display for the RLS objective funtion
        self.objfun_display_axis = fig.add_axes([0.1, 0.96, 0.8, 0.02])
        self.objfun_display_axis.imshow(mmc.compute_steepness_vector()[np.newaxis, :], cmap = plt.get_cmap("Oranges"))
        self.objfun_display_axis.set_aspect('auto')
        self.objfun_display_axis.set_yticklabels([])
        self.objfun_display_axis.yaxis.set_tick_params(size = 0)
    
    def onselect(self, verts):
        #Select a new working set
        self.path = Path(verts)
        self.selectedset = set(np.nonzero(self.path.contains_points(self.collection))[0])
        print('Selected ' + str(len(self.selectedset)) + ' points')
        newws = list(self.selectedset - self.lockedset)
        self.mmc.new_working_set(newws)
        self.redrawall()
    
    def onkeypressed(self, event):
        print('You pressed', event.key)
        if event.key == '1':
            print('Assigned all selected points to class 1')
            newclazz = 1
            mmc.claim_all_points_in_working_set(newclazz)
        if event.key == '0':
            print('Assigned all selected points to class 0')
            newclazz = 0
            mmc.claim_all_points_in_working_set(newclazz)
        if event.key == 'a':
            print('Selected all points')
            newws = list(set(range(len(self.collection))) - self.lockedset)
            self.mmc.new_working_set(newws)
            self.lasso.line.set_visible(False)
        if event.key == 'c':
            changecount = mmc.cyclic_descent_in_working_set()
            print('Performed ', changecount, 'cyclic descent steps')
        if event.key == 'l':
            print('Locked the class labels of selected points')
            self.lockedset = self.lockedset | self.selectedset
            newws = list(self.selectedset - self.lockedset)
            self.mmc.new_working_set(newws)
        if event.key == 'u':
            print('Unlocked the selected points')
            self.lockedset = self.lockedset - self.selectedset
            newws = list(self.selectedset - self.lockedset)
            self.mmc.new_working_set(newws)
        self.redrawall()
    
    def redrawall(self):
        #Color all class one labeled pixels red 
        oneclazz = np.nonzero(self.mmc.classvec)[0]
        col_row = self.collection[oneclazz]
        rowcs, colcs = col_row[:, 1], col_row[:, 0]
        red = np.array([255, 0, 0])
        for i in range(-self.windowsize, self.windowsize + 1):
            for j in range(-self.windowsize, self.windowsize + 1):
                self.img[rowcs+i, colcs+j, :] = red
        
        #Return the original color of the class zero labeled pixels 
        zeroclazz = np.nonzero(self.mmc.classvec - 1)[0]
        col_row = self.collection[zeroclazz]
        rowcs, colcs = col_row[:, 1], col_row[:, 0]
        for i in range(-self.windowsize, self.windowsize + 1):
            for j in range(-self.windowsize, self.windowsize + 1):
                self.img[rowcs+i, colcs+j, :] = self.img_orig[rowcs+i, colcs+j, :]
        self.imdata.set_data(self.img)
        
        #Update the slider position according to labeling of the current working set
        sliderval = 0
        if len(mmc.working_set) > 0:
            sliderval = len(np.nonzero(self.mmc.classvec_ws)[0]) / len(mmc.working_set)
        self.in_selection_slider.set_val(sliderval)
        
        #Update the RLS objective function display
        self.objfun_display_axis.imshow(mmc.compute_steepness_vector()[np.newaxis, :], cmap=plt.get_cmap("Oranges"))
        self.objfun_display_axis.set_aspect('auto')
        
        #Final stuff
        self.lasso.canvas.draw_idle()
        plt.draw()
        print_instructions()



def print_instructions():
    print()
    print('Draw a selection by holding down the left mouse button')
    print('Press the Fraction slider with the left mouse button to claim points inside the selection')
    print('Press a to select all points in the figure')
    print('Press 1 to claim all points in selection into class 1')
    print('Press 0 to claim all points in selection into class 0')
    print('Press l to lock all selected points to their current classes (e.g. they can not be claimed)')
    print('Press u to unlock all selected points after which they can be claimed again')
    print('Press c to perform a cyclic descent in the selection')
    print()


selector = SelectFromCollection(plt.figure(), mmc, img, pcoll, windowsize = windowsize)

plt.draw()
       
selector.redrawall()
print('All points are selected')

plt.show(block=True)

