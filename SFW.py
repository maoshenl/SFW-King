import numpy as np
import matplotlib.pyplot as plt
import random as rand
import scipy.special as ss
import time
import scipy.optimize

from scipy import integrate

#SWF stellar distribution function
def f(x, vr, vt, param=[2.0, -5.3, 2.5, 0.16, 1.5, -9.0, 6.9, 0.086, 21.0, 1.5, 1, 3, 1]):

        a,d,e, Ec, rlim, b, q, Jb, Vmax, rmax, alpha, beta, gamma = param

        rs = rmax/2.16           # rmax=2.16*rs
        Ps = (Vmax/0.465)**2.0   # Vmax=0.465*sqrt(Ps)
        Pr = genphi(x, alpha, beta, gamma, Vmax)
                                #Ps * ( 1 - (np.log(1+x))/x )
        J  = abs(x*rs * vt)           #J = v * r*sin(theta)
        E  = (vt*vt + vr*vr)/2.0 + Pr # v*v/2 + Pr

        Ec   = Ec * Ps
        xlim = rlim / rs #turn rlim in unit of rs
        Plim = genphi(xlim, alpha, beta, gamma, Vmax, rmax)
                #Ps * ( 1 - (np.log(1+xlim))/xlim ) #0.45*Ps
        Jb   = Jb * rs * (Ps**0.5) #*0.086

        if b <= 0:
                gJ = 1.0/(1 + (J/Jb)**-b)
        else:
                gJ = 1 + (J/Jb)**b

        N  = 1.0*10**3
        if E < Plim and E >= 0:
                hE = N*(E**a) * ((E**q + Ec**q)**(d/q)) * ((Plim - E)**e)
        else:
                hE = 0.0

        return hE * gJ


#model probility function
def fprob(x, vr, vt, param=[2.0, -5.3, 2.5, 0.16, 1.5, -9.0, 6.9, 0.086, 21.0, 1.5, 1,3,1]):
        return x*x*vt*f(x, vr, vt, param)


##define the general potential
def genphi(x, alpha = 1, beta = 3, gamma = 1, Vmax = 21.0, rmax = 1.5):
        rs = rmax/2.16
        Ps = (Vmax/0.465)**2.0

        x0 = 10**-12
        p1a = ss.hyp2f1((3-gamma)/alpha, (beta-gamma)/alpha, (3+alpha-gamma)/alpha, -x0**alpha)
        p1b = ss.hyp2f1((3-gamma)/alpha, (beta-gamma)/alpha, (3+alpha-gamma)/alpha,  -x**alpha)
        I1  = ( x0**(3-gamma) * p1a - x**(3-gamma) * p1b ) / (x * (gamma - 3))

        p2  = ss.hyp2f1( (-2+beta)/alpha, (beta-gamma)/alpha, (-2+alpha+beta)/alpha, -x**(-alpha))
        I2  = x**(2-beta) * p2 / (beta -2)
        ans1 = Ps * ( 1 - (I1+I2) )

        #to calculate the factor that gives the same mass within 300pc
        #factor = rhos_correct(.3, alpha, beta, gamma, Vmax, rmax)

        #ans2 = Ps * ( 1 - (np.log(1+x))/x )
        return ans1 #* factor


def rhos_correct(auxr = .3, alpha = 1, beta = 3, gamma = 1, Vmax = 21.0, rmax=1.5):
        rs = rmax/2.16
        Ps = (Vmax/0.465)**2.0

        x0 = 10**-12
        p1a = ss.hyp2f1((3-gamma)/alpha, (beta-gamma)/alpha, (3+alpha-gamma)/alpha, -x0**alpha)

        #to calculate the factor that gives the same mass within 300pc
        auxx = auxr/rs
        alpha0, beta0, gamma0 = [1,3,1]
        p1_a = ss.hyp2f1((3-gamma0)/alpha0, (beta0-gamma0)/alpha0, (3+alpha0-gamma0)/alpha0, -x0**alpha0)
        p1_b = ss.hyp2f1((3-gamma0)/alpha0,(beta0-gamma0)/alpha0,(3+alpha0-gamma0)/alpha0, -auxx**alpha0)
        auxmass0 = ( x0**(3-gamma0) * p1_a - auxx**(3-gamma0) * p1_b ) / ((gamma0 - 3))

        p1_b2 = ss.hyp2f1((3-gamma)/alpha,(beta-gamma)/alpha,(3+alpha-gamma)/alpha, -auxx**alpha)
        auxmass  = ( x0**(3-gamma) * p1a - auxx**(3-gamma) * p1_b2 ) / ((gamma - 3))
        factor = auxmass0/auxmass

        return factor


#get escape velocity; param = [rlim, Vmax, rmax, alpha, beta, gamma]
def vesc(x, param = [1.5, 21.0, 1.5, 1,3,1]): #pop1 MR, pop2 MP

        rlim, Vmax, rmax, alpha, beta, gamma = param

        rs = rmax/2.16           # rmax=2.16*rs
        Ps = (Vmax/0.465)**2.0   # Vmax=0.465*sqrt(Ps)

        xlim = rlim/rs           #turn rlim in unit of rs
        #Pr   = Ps * ( 1 - (np.log(1+x))/x )
        #Plim = Ps * ( 1 - (np.log(1+xlim))/xlim ) #lms 0.45*Ps
        Pr = genphi(x, alpha, beta, gamma, Vmax)
        Plim = genphi(xlim, alpha, beta, gamma, Vmax)
        vesc = (2 * (Plim - Pr))**0.5

        return vesc


#evaluate probility function at each step, to be used to create a proposal density function g(x,vr,vt)
def fbdvalues(param=[2.0, -5.3, 2.5, 0.16, 1.5, -9.0, 6.9, 0.086, 21.0, 1.5,1,3,1], steps = 10):
        xi  = 10**-12
        vri = 0
        vti = 0

        a,d,e, Ec, rlim, b, q, Jb, Vmax, rmax, alpha, beta, gamma = param

        rs    = rmax/2.16       # rmax=2.16*rs
	xlim = rlim / rs
        x0   = 10**-12
        vmax = vesc(x0, [rlim, Vmax, rmax, alpha, beta, gamma])

        xarr  = np.linspace(x0, xlim, steps)
        vrarr = np.linspace(vri, vmax, steps)
        vtarr = np.linspace(vti, vmax, steps)
	
	xxx, vrr, vtt = np.meshgrid(xarr, vrarr, vtarr, indexing='ij')


	#fillup the values.
        fvalues = np.zeros((steps, steps, steps))
        i = 0
        while i < steps:
                j = 0
                while j < steps:
                        k = 0
                        while k< steps:
                                fvalues[i,j,k] = 1.5*fprob(xxx[i,j,k], vrr[i,j,k], vtt[i, j, k], param)
                                k = k+1
                        j=j+1
                i=i+1

	#adding a constant value to prevent proposal distribution 
        #being zero while the target is nonzero
        fvalues = fvalues + np.amin(fvalues[fvalues>0])

        return xarr, vrarr, vtarr, fvalues

#evalue the CDF of the proposal distribution
def G(param, steps):
        xarr, vrarr, vtarr, p = fbdvalues(param, steps)
        a,d,e, Ec, rlim, b, q, Jb, Vmax, rmax, alpha, beta, gamma = param
        rs    = rmax/2.16
        xlim = rlim / rs
        x0   = 10**-12

        i = 1
        Gx = np.zeros(steps)
        Gxvr = []
        Gxvrvt = []
        pxvrvt = []
        while i < steps:
                Gvr = np.zeros(steps)
                Gvrvt = []
                pvrvt = []
                j = 1
                while j < steps:
                        Gvt = np.zeros(steps)
                        pvt = []
                        k = 1
                        while k < steps:
                                gijk = max(p[i-1,j-1,k-1], p[i-1,j-1,k], p[i-1,j,k-1], p[i-1,j,k],
                                           p[i  ,j-1,k-1], p[i  ,j-1,k], p[i  ,j,k-1], p[i  ,j,k] )
                                gvt = gijk * (vtarr[k] - vtarr[k-1])
                                Gvt[k] = Gvt[k-1] + gvt
                                pvt.append(gijk)
                                k = k+1

                        Gvrvt.append(Gvt/(max(Gvt)+10**-13))
                        pvrvt.append(pvt)
                        Gvr[j] = Gvr[j-1] + max(Gvt) * (vrarr[j] - vrarr[j-1])
                        j=j+1

                Gxvrvt.append(Gvrvt)
                pxvrvt.append(pvrvt)
                Gxvr.append(Gvr/(max(Gvr)+10**-13))
                Gx[i] = Gx[i-1] + max(Gvr)*(xarr[i]-xarr[i-1])
                i=i+1
        Gx = Gx/(max(Gx)+10**-13)
        return xarr, vrarr, vtarr, Gx, Gxvr, Gxvrvt, pxvrvt  #size steps+1 array, or list of that array,
                                                             #or list of such list


#to sample from the proposed distribution
def sampleg(xarr, vrarr, vtarr, Gx, Gxvr, Gxvrvt, pxvrvt):
        ux  = rand.random()
        uvr = rand.random()
        uvt = rand.random()

        xupindex = sum(Gx < ux*max(Gx))
        xloindex = xupindex - 1
        xi = xarr[xloindex] + rand.random() * (xarr[xupindex] - xarr[xloindex])

        Gvr = Gxvr[xloindex]
        vrupindex = sum(Gvr < uvr*max(Gvr))
        vrloindex = vrupindex - 1
        vri = vrarr[vrloindex] + rand.random() * (vrarr[vrupindex] - vrarr[vrloindex])

        Gvt = Gxvrvt[xloindex][vrloindex]
        vtupindex = sum(Gvt < uvt*max(Gvt))
        vtloindex = vtupindex - 1
        #print 'lo, hi, len(vtarr): ', vtloindex, vtupindex, len(vtarr)
        vti = vtarr[vtloindex] + rand.random() * (vtarr[vtupindex] - vtarr[vtloindex])

        gi = pxvrvt[xloindex][vrloindex][vtloindex]

        return xi, vri, vti, gi



#using importance sampling. Resamplefactor is a multiple of samplesize, which determines the number of
#discret points to evaluate and from which samples are drawn from
def sample(param, steps, samplesize, resamplefactor = 1.5):
        a,d,e, Ec, rlim, b, q, Jb, Vmax, rmax, alpha, beta, gamma = param
        rs    = rmax/2.16
        xarr, vrarr, vtarr, Gx, Gxvr, Gxvrvt, pxvrvt = G(param, steps)

        auxN = resamplefactor * samplesize #tune it
        auxlist = []
        weightlist = []
        m = 0
        while m < auxN:
                xi, vri, vti, gi = sampleg(xarr, vrarr, vtarr, Gx, Gxvr, Gxvrvt, pxvrvt)
                fi = fprob(xi, vri, vti, param)
                weightlist.append(fi/gi)
                auxlist.append([xi, vri, vti])
                m = m + 1

        samplelist = resample(weightlist, auxlist, samplesize)
        return convert(samplelist, rs)

#resample calculated points based on the weight
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


#give a random direction to radius, and returns x, y, z coordinate
#tranform the velocity coordinate to the spatial coordinate
def convert(samplelist, rs):
        samplelist = np.asarray(samplelist)
        xarr  = samplelist[:,0]
        vrarr = samplelist[:,1]
        vtarr = samplelist[:,2]

        r  = xarr * rs
        v  = (vrarr*vrarr + vtarr*vtarr )**0.5

        rsize = len(r)
        ru    = np.random.random_sample( rsize )
        theta = np.arccos(1-2*ru) #inverse sampling the distribution for theta
        phi   = np.random.random_sample( rsize ) * 2.0 * np.pi

        vsign = np.sign( np.random.random_sample( rsize ) - 0.5 )
        vphi  = np.random.random_sample( rsize ) * 2.0 * np.pi

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        vz2 = vsign * vrarr
        vx2 = vtarr * np.cos(vphi)
        vy2 = vtarr * np.sin(vphi)

        #passive rotation, using rotation matrix 
        #to rotate the zhat of calculate velocity into the zhat of the spatial coordinate
        vx = np.cos(theta)*np.cos(phi)*vx2 - np.sin(phi)*vy2 + np.sin(theta)*np.cos(phi)*vz2
        vy = np.cos(theta)*np.sin(phi)*vx2 + np.cos(phi)*vy2 + np.sin(theta)*np.sin(phi)*vz2
        vz = -np.sin(theta)*vx2 + np.cos(theta)*vz2
        return x, y, z, vx, vy, vz


#use accept and reject sampling technique
def accept_reject_sample( param = [2.0, -5.3, 2.5, 0.16, 1.5, -9.0, 6.9, 0.086, 21.0, 1.5, 1,3,1], samplesize = 3000):

        a,d,e, Ec, rlim, b, q, Jb, Vmax, rmax, alpha, beta, gamma = param
        rs    = rmax/2.16       # rmax=2.16*rs

        #xm0, vrm0, vtm0, fmax0 = findmax(param)
        xm,vrm, vtm = scipy.optimize.fmin(lambda (x,vr,vt): -fprob(x,vr,vt, param), (0.1,1,1), maxiter=999999)
        fmax1 = 0 #*fmax0
        fmax2 = 1.1*fprob(xm, vrm, vtm, param)
        fmax  = max([fmax1, fmax2])
        #print "fmax: ", fmax1, fmax2

        samplelist = []
        num = 1
        aux = 1
        while (num <= samplesize):

                x = rlim * rand.random() / rs
                x0 = 10**-8
                vmax0 = vesc(x0, [rlim, Vmax, rmax, alpha, beta, gamma])
                vr    = vmax0 * rand.random()
                vt    = vmax0 * rand.random()

                u  = rand.random() #new *fmax 
                fi = fprob(x, vr, vt, param)
                gi = fmax
                if (fi/gi >= u): #new
                        samplelist.append( [x,vr,vt] )
                        num = num + 1
                if (fi/gi > 1.0):
                        print num, 'Warning old: f(x)/g(x) > 1', fi, gi

                aux = aux + 1
        print "acceptance rate: ", num/(aux+0.00001)
        return convert(samplelist, rs)


#param = [2.0, -5.3, 2.5, 0.16, 1.5, -9.0, 6.9, 0.086, 21.0, 1.5,1,3,1]
#x2,y2,z2, vx2,vy2,vz2 = accept_reject_sample(param, 2000)
#print x2
