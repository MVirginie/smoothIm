# -*- coding: utf-8 -*-

import numpy as np
#from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import matplotlib as mp
from mpl_toolkits.mplot3d import Axes3D
import sys
import scipy as sp

class Denoising_Methods :
    """
     Class with all the data needed to solve the scheme
	  - dt         : the time's step
	  - dx         : the x's step
      - dy         :
      - nb_it      : the number of iterations
      - init_cond  : the initial condition (t=0)
      - bound_cond : the boundaries conditions (j= 0 && j = N-1)
	  - t_end      : the final time
      """

    def __init__(self, dt, dx, dy,nb_it) :
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.nb_it = nb_it
        self.fig = plt.figure()
        count = 1
        t_temp = 0.
        t = [0.]
        while count < self.nb_it :
            t_temp = t_temp+ self.dt
            t = t+[t_temp]
            count = count +1
        self.t = t #The mesh --> t=[0,dt,2dt...]
        print("dt : %.2f"%self.dt)


    def boundaries_cond(self, im, x, y, type):
        """
        Define the boundaries conditions, i.e extend the image by symmetry
        """
        # CASE HEAT EQUATION
        im = np.c_[im[:,0], im] #add the first column at the beginning of the image
        im = np.c_[im, im[:,y]] # add a column at the end
        im = np.r_['0,2',im[0,:], im] #add the first line at the beginning of the image
        im = np.r_['0,2',im, im[x,:]]
        #CASE PERONA MALIK
        if type == 2 :
            im = np.c_[im[:,0],im]
            im = np.c_[im, im[:,y]]
            im = np.r_['0,2',im[0,:], im]
            im = np.r_['0,2',im, im[x,:]]
        return im

    def heat_equation(self,im):
        """ Calculate the approximation of the heat equation
            w\ the finite differences method
            (euler_explicit)
        """
        #Define the figure to plot
        #self.fig.clear()
        ax1 = self.fig.add_subplot(1,2,1)
        ax1.imshow(im, cmap = 'gray')
        ax1.set_title("Image without blur ")
        ax = self.fig.add_subplot(122)
        ax.imshow(im, cmap = 'gray')
        ax.set_title("Image without blur ")
        (x,y) = np.shape(im)
        u = np.ndarray(shape = (x+2,y+2)) #Create a new array
        im = self.boundaries_cond(im,x,y,1)

        for n in range(1,len(self.t)+1):
            u_xp1 = im[2:x+1,1:y]
            u_xm1 = im[0:x-1,1:y]
            u_yp1 =im[1:x,2:y+1]
            u_ym1 = im[1:x,0:y-1]
            u[1:x,1:y] = im[1:x,1:y]+np.multiply(self.dt,(u_xp1+u_xm1+u_ym1+u_yp1-4.*im[1:x,1:y]))
            im[1:x, 1:y] = u[1:x,1:y]
            im[:,0] = u[:,1]
            im[:,y+1] = u[:,y]
            im[0,:] =u[1,:]
            im[x+1,:]= u[x,:]
            plt.title("Heat equation applied %.2f times"%n)
            ax.imshow(im[1:x,1:y], cmap = 'gray')
            plt.pause(0.05)
            ax.cla()

        ax.imshow(im[1:x,1:y], cmap = 'gray')
        ax.set_title("Heat equation applied %.1f times"%len(self.t))
        plt.show()
        return self.fig

    def c_function(self, alpha, x):
        """
        Define the function needed to apply Perona Malik
        """
        return 1/(1+(x/alpha)**2)




    def perona_malik(self, im, alpha, fun):
        """
        Apply the perona malik filter on the picture (im)
        """
        p = np.arange(0,3,1) #discretization of the gradient
        ax1 = self.fig.add_subplot(1,2,1)
        ax = self.fig.add_subplot(1,2,2)
        ax1.imshow(im, cmap = 'gray')
        ax1.set_title("Image ")
        ax.imshow(im, cmap = 'gray')
        im = np.asarray(im)
        (x,y) = np.shape(im)
        im = self.boundaries_cond(im,x,y,2)
        u = np.ndarray(shape = (x+4,y+4), dtype = 'double')

        for n in range(1,len(self.t)+1):
            ax.cla()
            c_grad = fun(alpha,self.opti_grad(im, x,y, fun, alpha))
            grad = np.ndarray(shape = (x+4,y+4,4), dtype = 'double')
            grad[:,:,0] = np.subtract(np.roll(im, 1, axis = 0), im) #grad_x
            grad[:,:,1] = np.subtract(np.roll(im, 1, axis = 1), im) #grad_y
            grad[:,:,2] = np.subtract(im, np.roll(im, -1, axis = 0)) #grad_x-
            grad[:,:,3] = np.subtract(im, np.roll(im, -1, axis = 1)) #grad_y-
            shift = np.ndarray(shape = (x+4,y+4,2), dtype = 'double')
            shift[:,:,0]=np.roll(c_grad, -1, axis = 0)
            shift[:,:,1]= np.roll(c_grad, -1, axis = 1)

            u = im+self.dt*(np.multiply(c_grad,grad[:,:,0])-np.multiply(shift[:,:,0],grad[:,:,2])+np.multiply(c_grad,grad[:,:,1])-np.multiply(shift[:,:,1],grad[:,:,3]))

            im[2:x+1, 2:y+1] = u[2:x+1,2:y+1]
            im[:,0] = u[:,3]
            im[:,1]= u[:,2]
            im[:,y+1] = u[:,y]
            im[:,y+2]=u[:,y-1]
            im[0,:] =u[3,:]
            im[1,:] = u[2,:]
            im[x+1,:]= u[x,:]
            im[x+2,:]= u[x-1,:]
            plt.title("Perona-Malik applied %.2f times"%n)
            ax.imshow(im[2:x+1, 2:y+1], cmap = 'gray')
            plt.pause(0.001)
        ax.imshow(im[2:x+1, 2:y+1], cmap = 'gray')
        plt.title("P-M equation applied %.1f times"%len(self.t))
        return self.fig

    def opti_grad(self, im, x,y, fun, alpha) :

        """
	Calculate |g| with g, the gradient \ optimisation
        """
        grad_test = np.ndarray(shape = (x+4,y+4,4), dtype = 'double')
        grad_test[:,:,0] = np.subtract(np.roll(im, 1, axis = 0), im) #grad_x
        grad_test[:,:,1] = np.subtract(np.roll(im, 1, axis = 1), im) #grad_y
        grad_test[:,:,2] = np.subtract(im, np.roll(im, -1, axis = 0)) #grad_x-
        grad_test[:,:,3] = np.subtract(im, np.roll(im, -1, axis = 1)) #grad_y-

        max_min =  np.ndarray(shape = (x+4,y+4,4), dtype = 'double')
        max_min[:,:,0:2] = np.fmin(grad_test[:,:,0:2], 0.0)
        max_min[:,:,2:4] = np.fmax(grad_test[:,:,2:4],0.0)

        max_min[:,:,0:4] = np.multiply(max_min[:,:,0:4],max_min[:,:,0:4])


        g_final = np.ndarray(shape = (x+4,y+4), dtype = 'double')
        g_final = np.sqrt(max_min[:,:,0]+max_min[:,:,1]+max_min[:,:,2]+max_min[:,:,3])

        return g_final
