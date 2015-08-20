import numpy as np
import random as rand
import scipy.optimize

from scipy import integrate
from scipy import special
#from scipy.stats import multivariate_normal
from math import erf


#density function of kingmodels
def f(x, v, psi0, ratio0, xarr, psisigma2):

        sigma2= (psi0/ratio0)
        E = psisigma2[ np.abs(xarr-x).argmin() ] - v*v/(2.*sigma2)

        rho1 = 1.
        #r0  = 1.
        if E>0:
                DF = rho1*(2*np.pi*sigma2)**(-1.5) * (np.exp(E) - 1)
        else:
                DF = 0.0

        return DF

#numerically evaluate the reduced potential/sigma^2 as a function of radius
def getPsigma2( psi0=100., ratio=3.0):
        sigma2 = (psi0/ratio)
        rho0_rho1 = (np.exp(ratio)*erf(ratio**0.5) - ((4*ratio/np.pi)**0.5)*(1+2*ratio/3.))


        def deriv(u,x):    #r in kpc, x = r/r0
            u0 = u[0]
            u1 = u[1]
            #if  u0<0:  #need to investigate negative values
            #    print x, u0
            #    u0 = 0
            u1prime =(  (-u1*2.0/x) -
                        (9./(rho0_rho1)) * (np.exp(u0)*erf(u0**0.5)  - ((4*u0/np.pi)**0.5)*(1+2*u0/3.)) )
            u0prime = u1
            uprime  = np.array([u0prime,u1prime])

            return uprime

        x  = np.linspace(0.00001, 800, 100000)  #
        uinitial = np.asarray([psi0/sigma2, 0])

        psi_sigma2 = integrate.odeint(deriv, uinitial, x)[:,0] #[:,1] contains the values of psi_prime

        #print 'max', max(x[psi_sigma2>0])
        return x[psi_sigma2>0], psi_sigma2[psi_sigma2>0]


#probability of x and v. 
def fprob(x, v, psi0, ratio0, xarr, psisigma2):
        r0 = 1.
        return f(x, v, psi0, ratio0, xarr, psisigma2) * v*v*x*x *(4.*np.pi)**2


#evaluate the function at discrete points to construct a step function as a proposal distribution
def fbdvalues(psi0, ratio0, xarr, psisigma2, steps = 10):

        sigma2 = (psi0/ratio0)
        P    = psisigma2[ np.abs(xarr-xarr[0]).argmin() ]
        vmax = (2.0*P*sigma2)**0.5

        xarray = np.linspace(0, max(xarr), steps)
        varray = np.linspace(0, vmax, steps) 

        xx, vv = np.meshgrid(xarray, varray, indexing='ij')
        fvalues = np.zeros((steps, steps))
        i = 0
        while i < steps:
                j = 0
                while j < steps:
                        fvalues[i,j] = 2.0*fprob(xx[i,j], vv[i,j], psi0, ratio0, xarr, psisigma2)
                        j = j+1
                i = i+1

        #(optional) adding a constant to prevent proposal distribution 
        #being zero while the target is nonzero
        fvalues = fvalues + np.amin(fvalues[fvalues>0])

        return xarray, varray, fvalues

#evalue CDF of the proposal distribution
def G(psi0, ratio0, xarr, psisigma2, steps = 10):
        xarray, varray, p = fbdvalues(psi0, ratio0, xarr, psisigma2, steps)
        
        i = 1
        Gx = np.zeros(steps)
        Gxv = []
        pxv = []
        while i < steps:
                Gv = np.zeros(steps)
                pv = []
                j = 1
                while j < steps:
                        gij = max(p[i-1,j-1], p[i-1,j], p[i,j-1], p[i,j]) #1.1 * fm
                        Gv[j] = Gv[j-1] + gij * (varray[j] - varray[j-1])
                        pv.append(gij)
                        j = j+1
                Gxv.append(Gv/(max(Gv)+10**-12))
                pxv.append(pv)
                Gx[i] = Gx[i-1] + max(Gv)*(xarray[i] - xarray[i-1])
                i = i+1

        Gx = Gx/(max(Gx)+10**-12)
        return xarray, varray, Gx, Gxv, pxv

#sample from the proposal distribution
def sampleg(xarray, varray, Gx, Gxv, pxv):
        ux = rand.random()
        uv = rand.random()

        xupindex = sum(Gx < ux*max(Gx))
        xloindex = xupindex-1
        xi = xarray[xloindex] + rand.random() * (xarray[xupindex] - xarray[xloindex])

        Gv = Gxv[xloindex]
        vupindex = sum(Gv < uv*max(Gv))
        vloindex = vupindex - 1
        vi = varray[vloindex] + rand.random() * (varray[vupindex] - varray[vloindex])

        gi = pxv[xloindex][vloindex]

        return xi, vi, gi


#use importance sampling to sample king models
def sample(sigma, ratio0=3.0, steps=10, samplesize=1000, resamplefactor = 1.5):

        sigma2 = sigma**2
        psi0   = sigma2 * ratio0
        xarr, psisigma2 = getPsigma2(psi0, ratio0)
        xmax   = max(xarr) #aka tidal radius
        sigma2 = (psi0/ratio0)


        #sample using step function G as proposed density
        xarray, varray, Gx, Gxv, pxv = G(psi0, ratio0, xarr, psisigma2, steps)
        samplelist = []
        auxN = resamplefactor * samplesize
        auxlist = []
        weightlist = []
        m = 0
        while m < auxN:
                xi, vi, gi = sampleg(xarray, varray, Gx, Gxv, pxv)
                auxlist.append( [xi,vi] )
                fi = fprob(xi, vi, psi0, ratio0, xarr, psisigma2)
                weightlist.append(fi/gi)
                if ( fi > 0 and gi == 0 ):
                        print "Warning: f/g is infinity, try increasing the number of steps."
                m = m + 1

        samplelist = resample(weightlist, auxlist, samplesize)

        return convert(samplelist)


#sampling the calculated points based on the importance weight. 
def resample(weightlist, auxlist, samplesize):
        samplelist = []
        uarray = np.random.random_sample( samplesize ) * sum(weightlist)
        uarray = uarray[np.argsort(uarray)]
        Warray = np.zeros(len(weightlist)+1)
        upto = 0
        j = 0
        for i, wi in enumerate(weightlist):
                while j < samplesize:
                        if upto + wi > uarray[j]:
                                samplelist.append(auxlist[i])
                                j = j+1
                        else:
                                break
                upto += wi
        return np.array(samplelist)

#give random direction to both velocity and radial position, convert to cartesian coordinate
def convert(samplelist):
        #into x, y, z, vx, vy, vz
        samplelist = np.asarray(samplelist)
        x = samplelist[:,0]
        rsize = len(x)
        ru = np.random.random_sample( rsize )
        rtheta = np.arccos(1-2*ru)
        rphi   = np.random.random_sample( rsize ) * 2.0 * np.pi
        for i in xrange(rsize):
                rphi[i] = rand.random() * 2 * np.pi

        xx = x * np.sin(rtheta) * np.cos(rphi)
        yy = x * np.sin(rtheta) * np.sin(rphi)
        zz = x * np.cos(rtheta)

        v = samplelist[:,1]
        vu     = np.random.random_sample( rsize )
        vtheta = np.arccos(1-2*vu)
        vphi   = np.random.random_sample( rsize ) * 2.0 * np.pi
        vx     = v * np.sin(vtheta) * np.cos(vphi)
        vy     = v * np.sin(vtheta) * np.sin(vphi)
        vz     = v * np.cos(vtheta)

        return xx, yy, zz, vx,vy,vz


