import numpy as np
import bisect
import copy
import matplotlib.pyplot as plt
import random
RNG = random.SystemRandom()

EPSILON = 1e-6
INF = np.infty

PRINTCASE = True

## maximum number of grids allowed.
DEFAULT_NUM_GRID = 50

## if len(grid) > GRID_THRESHOLD,
## try re-interpolating to be PLP first.
GRID_THRESHOLD = 8


########################################################################################
#                                    CLASS RAMP
########################################################################################

class Ramp():
    """
    Ramp is a trajectory segment with constant acceleration.
    
    x0       : initial value
    v        : initial velocity
    finalv   : final velocity
    a        : acceleration
    T        : duration
    distance : x1 - x0
    """
    def __init__(self, v, a, T, x0 = 0):
        assert(T > -EPSILON)

        self.x0 = float(x0) # initial condition of the ramp
        self.v = float(v)
        self.a = float(a)
        self.T = float(T)
        self.finalv = self.v + self.a*self.T
        self.distance = 0.5*(self.v + self.finalv)*self.T


    ## needed for merge rampslist
    def UpdateDuration(self, newT):
        assert(newT > -EPSILON)

        self.T = newT
        self.finalv = self.v + self.a*self.T
        self.distance = 0.5*(self.v + self.finalv)*self.T
        
        
    def Eval(self, t):
        assert(t > -EPSILON)
        assert(t < self.T + EPSILON)
        
        v1 = self.Evald(t)
        return 0.5*(self.v + v1)*t + self.x0
    
    
    def Evald(self, t):
        assert(t > -EPSILON)
        assert(t < self.T + EPSILON)

        return self.v + self.a*t

    
    def Evaldd(self, t):
        assert(t > -EPSILON)
        assert(t < self.T + EPSILON)

        return self.a
## end class Ramp


########################################################################################
#                                  CLASS RAMPSLIST
########################################################################################

class RampsList():
    """
    RampsList consists of ramps concatenated together.
    The class is initialized from a list of ramps (rampslist).
    
    rampslist : list of ramps
    isvoid    : parameter indicating whether self.rampslist is empty
    x0        : initial value
    duration  : total duration
    distance  : x1 - x0
    cumulateddurationslist : list containing time instants at the begining of each ramp
    cumulateddistanceslist : list containing initial conditions of each ramp    
    """
    
    def __init__(self, rampslist = []):
        self.cumulateddurationslist = []
        self.cumulateddistanceslist = []
        t = 0.0
        d = 0.0
        
        if (len(rampslist) == 0):
            self.rampslist = []
            self.isvoid = True
            self.x0 = 0
        else:
            self.rampslist = rampslist[:]
            self.isvoid = False
            self.SetInitialValue(rampslist[0].x0)
            for i in range(len(self.rampslist)):
                self.cumulateddurationslist.append(t)
                t += self.rampslist[i].T
                self.cumulateddistanceslist.append(d)
                d += self.rampslist[i].distance                
        self.duration = t
        self.distance = d
                

    def __getitem__(self, i):
        return self.rampslist[i]
    

    def __len__(self):
        return len(self.rampslist)


    def SetInitialValue(self, x0):
        self.x0 = x0
        self.rampslist[0].x0 = x0
        for i in range(1, len(self.rampslist)):
            self.rampslist[i].x0 = (self.rampslist[i - 1].x0 + 
                                    self.rampslist[i - 1].distance)
        

    def Append(self, rampslist0):
        if (self.isvoid):
            if (len(rampslist0) > 0):
                self.isvoid = False
                self.rampslist = rampslist0[:]
                self.SetInitialValue(rampslist0[0].x0)
                self.cumulateddurationslist = rampslist0.cumulateddurationslist[:]
                self.cumulateddistanceslist = rampslist0.cumulateddistanceslist[:]
                self.duration = rampslist0.duration
                self.distance = rampslist0.distance
        else:
            dur = self.duration
            dist = self.distance
            for i in range(len(rampslist0)):
                self.rampslist.append(rampslist0[i])
                self.cumulateddurationslist.append(dur)
                self.cumulateddistanceslist.append(dist)
                dur += rampslist0[i].T
                dist += rampslist0[i].distance
                self.rampslist[-1].x0 = (self.rampslist[-2].x0 + 
                                         self.rampslist[-2].distance)
            self.duration = dur
            self.distance = dist


    ## merge consecutive ramps together if they have the same acceleration
    def Merge(self):
        if not (self.isvoid):
            cura = self.rampslist[0].a
            nmerged = 0 ## count the number of merged ramps
            for i in range(1, len(self.rampslist)):
                j = i - nmerged
                if (abs(self.rampslist[j].a - cura) < EPSILON):
                    ## merge rampslist
                    redundantramp = self.rampslist.pop(j)
                    newT = self.rampslist[j - 1].T + redundantramp.T
                    self.rampslist[j - 1].UpdateDuration(newT)

                    ## merge cumulateddurationslist
                    self.cumulateddurationslist.pop(j)
        
                    ## merge cumulateddistanceslist
                    self.cumulateddistanceslist.pop(j)

                    nmerged += 1
                else:
                    cura = self.rampslist[j].a
                    
            
    def FindRampIndex(self, t):
        assert(t > -EPSILON)
        assert(t < self.duration + EPSILON)

        if t == 0:
            i = 0
            remainder = 0.0
        else:
            i = bisect.bisect_left(self.cumulateddurationslist, t) - 1
            remainder = t - self.cumulateddurationslist[i]
        return i, remainder


    def Eval(self, t):
        assert(t > -EPSILON)
        assert(t < self.duration + EPSILON)

        i, remainder = self.FindRampIndex(t)
        return self.rampslist[i].Eval(remainder)
    

    def Evald(self, t):
        assert(t > -EPSILON)
        assert(t < self.duration + EPSILON)

        i, remainder = self.FindRampIndex(t)
        return self.rampslist[i].Evald(remainder)


    def Evaldd(self, t):
        assert(t > -EPSILON)
        assert(t < self.duration + EPSILON)

        i, remainder = self.FindRampIndex(t)
        return self.rampslist[i].a
    

    def Plot(self, fignum = None, color = 'g', dt = 0.01, lw = 2):
        tvect = np.arange(0.0, self.duration, dt)
        np.append(tvect, self.duration)
        xvect = [self.Eval(t) for t in tvect]
        if not (fignum == None):
            plt.figure(fignum)
        plt.plot(tvect, xvect, color, linewidth = lw)
        plt.show(False)        

        
    def Plotd(self, fignum = None, color = 'b', lw = 2):
        if (self.rampslist == None):
            pass
        else:
            vvect = [k.v for k in self.rampslist]
            vvect.append(self.rampslist[-1].finalv)
            
            tvect = self.cumulateddurationslist[:]
            tvect.append(self.duration)
            
            if not (fignum == None):
                plt.figure(fignum)
            plt.plot(tvect, vvect, color, linewidth = lw)
            plt.show(False)
        
        
    def Plotdd(self, fignum = None, color = 'm', lw = 2):
        avect = []
        for k in self.rampslist:
            avect.append(k.a)
            avect.append(k.a)
        tvect = []
        for i in range(len(self.cumulateddurationslist) - 1):
            tvect.append(self.cumulateddurationslist[i])
            tvect.append(self.cumulateddurationslist[i + 1])
        tvect.append(self.cumulateddurationslist[-1])
        tvect.append(self.duration)
        if not(fignum == None):
            plt.figure(fignum)
        plt.plot(tvect, avect, color, linewidth = lw)
        plt.show(False)
        

    def __str__(self):
        return "d = {0}\nv0 = {1}\nv1 = {2}".format(self.distance, self.rampslist[0].v,
                                                    self.rampslist[-1].finalv)
## end class RampsList


########################################################################################
#                                 CLASS RAMPSLISTND
########################################################################################

class RampsListND():
    """
    RampsListND is an n-dimensional rampslist
    The class is initialized from a list of rampslists
    
    rampslistnd   : list of rampslists
    isvoid        : parameter indicating whether self.rampslist is empty
    x0vect        : initial vector
    duration      : total duration
    distance      : x1vect - x0vect
    cumulateddurationslist : list containing time instants at the begining of each ramp
    cumulateddistanceslist : list containing initial conditions of each ramp    
    """
    def __init__(self, rampslistnd = []):
        if (len(rampslistnd) == 0):
            self.isvoid = True
            self.duration = 0
            self.rampslistnd = rampslistnd
            self.ndof = 0
            self.x0vect = 0
            self.switchpointslist = []
        
        else:
            ## check first if every rampslist in rampslistnd has the same duration
            dur = rampslistnd[0].duration
            for i in range(1, len(rampslistnd)):
                assert(abs(rampslistnd[i].duration - dur) < EPSILON)

            self.isvoid = False
            self.duration = dur
            self.rampslistnd = rampslistnd
            self.ndof = len(rampslistnd)
            self.x0vect = np.asarray([rampslistnd[i].x0 for i in range(self.ndof)])
            self.SetInitialValue(self.x0vect)
        
            ## create a list of all switch points
            ## the list include both end points
            switchpointslist = rampslistnd[0].cumulateddurationslist[1::]
            for i in range(1, self.ndof):
                for s in self.rampslistnd[i].cumulateddurationslist[1::]:
                    switchpointslist.insert(bisect.bisect_left(switchpointslist, s), s)

            self.switchpointslist = []
            if len(switchpointslist) > 0:
                self.switchpointslist.append(switchpointslist[0])
                ## remove redundant switching points
                for i in range(1, len(switchpointslist)):
                    if (abs(switchpointslist[i] - switchpointslist[i - 1]) > EPSILON):
                        self.switchpointslist.append(switchpointslist[i])
                
            ## the list also includes both end points
            self.switchpointslist.insert(0, 0)
            self.switchpointslist.append(self.duration)
        
        
    @staticmethod
    def FromChunksList(chunkslist):
        """          
        chunkslist = [chunk_0, chunk_1, chunk_2, ..., chunk_M]
        chunk_i = [rampslist_i0, rampslist_i1, ..., rampslist_iN]
        """
        ndof = len(chunkslist[0])
        rampslistnd = []
        for i in range(ndof):
            R = RampsList()
            for j in range(len(chunkslist)):
                R.Append(chunkslist[j][i])
            rampslistnd.append(R)            
        return RampsListND(rampslistnd)


    ## recover rampslistnd from 2nd-order TOPP trajectorystring
    @staticmethod
    def FromString(trajectorystring):
        """
        trajectorystring structure
                |  duration
                |  ndof
        chunk1  <  poly1
                |  poly2
                |  . . .
                |  polyndof
        """
        ss = trajectorystring.split()
        ndof = int(ss[1])
        len1chunk = ndof*3 + 2
        nchunks = len(ss)/len1chunk
        durationslist = [float(ss[i*len1chunk]) for i in range(nchunks)]
        rampslistnd = []
        for j in range(ndof):
            rampslist = RampsList()
            for i in range(nchunks):
                index = (i*len1chunk) + (3*j) + 2
                x = float(ss[index])
                v = float(ss[index + 1])
                a = 2*float(ss[index + 2])
                ramp = Ramp(v, a, durationslist[i], x0 = x)
                rampslist.Append(RampsList([ramp]))
            rampslist.Merge()
            rampslistnd.append(rampslist)
        return RampsListND(rampslistnd)        
        

    ## convert to trajectorystring compatible with TOPP
    def __str__(self):
        trajectorystring = ''
        separator1 = ''
        for swindex in range(len(self.switchpointslist) - 1):
            trajectorystring += separator1
            separator1 = '\n'
            
            dur = self.switchpointslist[swindex + 1] - self.switchpointslist[swindex]
            trajectorystring += str(dur)
            trajectorystring += '\n'
            trajectorystring += str(self.ndof)
            
            xvect = self.Eval(self.switchpointslist[swindex])
            vvect = self.Evald(self.switchpointslist[swindex])
            avect = self.Evaldd(self.switchpointslist[swindex] + EPSILON)
            for i in range(self.ndof):
                trajectorystring += '\n'
                polynomialcoeff = "{0} {1} {2}".format(xvect[i], vvect[i], 0.5*avect[i])
                trajectorystring += polynomialcoeff
                
        return trajectorystring

    
    def __getitem__(self, i):
        return self.rampslistnd[i]


    def __len__(self):
        return len(self.rampslistnd)


    def SetInitialValue(self, x0vect):
        assert(len(x0vect) == self.ndof)
        
        self.x0vect = np.asarray(x0vect)
        for i in range(self.ndof):
            self.rampslistnd[i].SetInitialValue(x0vect[i])

            
    def Append(self, rampslistnd0):
        if (self.isvoid):
            if (len(rampslistnd0) > 0):
                self.isvoid = False
                self.duration = rampslistnd0.duration
                self.rampslistnd = rampslistnd0[:]
                self.ndof = len(rampslistnd0)
                self.x0vect = np.asarray([rampslistnd0[i].x0 for i in range(self.ndof)])
                self.switchpointslist = rampslistnd0.switchpointslist[:]
        else:
            assert(self.ndof == rampslistnd0.ndof)
            
            originaldur = self.duration
            self.duration += rampslistnd0.duration
            for j in range(self.ndof):
                self.rampslistnd[j].Append(rampslistnd0[j])
            newswitchpointslist = rampslistnd0.switchpointslist[:]
            newswitchpointslist = [newswitchpointslist[i] + originaldur 
                                   for i in range(len(newswitchpointslist))]
            self.switchpointslist.pop()
            self.switchpointslist.extend(newswitchpointslist)            


    def Eval(self, t):
        q = []
        for i in range(self.ndof):
            q.append(self.rampslistnd[i].Eval(t))
        return np.asarray(q)
    
    
    def Evald(self, t):
        qd = []
        for i in range(self.ndof):
            qd.append(self.rampslistnd[i].Evald(t))
        return np.asarray(qd)


    def Evaldd(self, t):
        qdd = []
        for i in range(self.ndof):
            qdd.append(self.rampslistnd[i].Evaldd(t))
        return np.asarray(qdd)
    
    
    def Plot(self, fignum = None, dt = 0.005):
        if not (fignum == None):
            plt.figure(fignum)
        tvect = np.arange(0, self.duration, dt)
        np.append(tvect, self.duration)
        qvect = np.array([self.Eval(t) for t in tvect])
        plt.plot(tvect, qvect, 'x', linewidth = 2)
        handle = ["joint {0}".format(i) for i in range(self.ndof)]
        plt.legend(handle)


    def Plotd(self, fignum = None, includingsw = False):
        if not (fignum == None):
            plt.figure(fignum)
        tvect = self.switchpointslist # np.arange(0, self.duration, dt)
        np.append(tvect, self.duration)
        qdvect = np.array([self.Evald(t) for t in tvect])
        plt.plot(tvect, qdvect, linewidth = 2)
        handle = ["joint {0}".format(i) for i in range(self.ndof)]
        plt.legend(handle)
        
        if includingsw:
            ax = plt.gca().axis()
            for sw in self.switchpointslist:
                plt.plot([sw, sw], [ax[2], ax[3]], 'r', linewidth = 1)

    
    def Plotdd(self, fignum = None, includingsw = False):
        if not (fignum == None):
            plt.figure(fignum)
            
        for i in range(self.ndof):
            rampslist = self.rampslistnd[i]
            avect = []
            for r in rampslist:
                avect.append(r.a)
                avect.append(r.a)
            t = []
            for j in range(len(rampslist.cumulateddurationslist) - 1):
                t.append(rampslist.cumulateddurationslist[j])
                t.append(rampslist.cumulateddurationslist[j + 1])
            t.append(rampslist.cumulateddurationslist[-1])
            t.append(rampslist.duration)

            plt.plot(t, avect, linewidth = 2)
        
        handle = ["joint {0}".format(i) for i in range(self.ndof)]
        plt.legend(handle)
        
        if includingsw:
            ax = plt.gca().axis()
            for sw in self.switchpointslist:
                plt.plot([sw, sw], [ax[2], ax[3]], 'r', linewidth = 1)
## end class RampsListND


########################################################################################
#                                   CLASS INTERVAL
########################################################################################
EPSILON2 = 1e-10

class Interval():
    """
    class Interval : currently implemented as closed interval
    """
    def __init__(self, l = np.infty, u = -np.infty):
        if (l <= u):
            self.l = float(l)
            self.u = float(u)
            self.isvoid = False
        else:
            self.l = float(l)
            self.u = float(u)
            self.isvoid = True
            

    def contain(self, a):     
        if (self.l <= a) and (a <= self.u):
            return True
        else:
            return False

        
    def __str__(self):
        return "[{0}, {1}]".format(self.l, self.u)


## Intersect checks whether two given intervals intersect each other.
## If ret is True, then it returns the intersection.
def Intersect(i0, i1, ret = True):    
    if (i0.u >= i1.l - EPSILON2) and (i0.l <= i1.u + EPSILON2):
        if ret == True:
            return Interval(max(i0.l, i1.l), min(i0.u, i1.u))
        else:
            return True
    else:
        if ret == True:
            return Interval()
        else:
            return False


## SolveIneq solves an inequality of the form
##              l <= ax + b <= u.
def SolveIneq(a, b, l = -np.infty, u = np.infty):
    if (l > u):
        return Interval()
    lu = Interval(l, u)    
    if (a == 0.0) or (abs(a) < EPSILON2):
        if (Intersect(Interval(b, b), lu)):
            return lu
        else:
            return Interval()       
    l = l - b
    u = u - b
    if (a > 0):
        return Interval(l/a, u/a)
    else:
        return Interval(u/a, l/a)
## end class Interval


########################################################################################
#                              INTERPOLATION ROUTINES
########################################################################################

## Simultaneous interpolation with zero terminal velocities
def InterpolateZeroVelND(x0vect, x1vect, vmvect, amvect, delta = 0):
    ndof = len(x0vect)
    try:
        assert(ndof == len(x1vect))
        assert(ndof == len(vmvect))
        assert(ndof == len(amvect))
    except AssertionError:
        print 'x0vect', x0vect
        print 'x1vect', x1vect
        print 'vmvect', vmvect
        print 'amvect', amvect
        raw_input()
        
    rampslistnd0 = [RampsList() for j in range(ndof)]
    vmin = INF
    amin = INF
    for j in range(ndof):
        if (abs(x1vect[j] - x0vect[j]) > EPSILON):
            vmin = min(vmin, vmvect[j]/(abs(x1vect[j] - x0vect[j])))
            amin = min(amin, amvect[j]/(abs(x1vect[j] - x0vect[j])))
    
    assert(vmin < INF)
    assert(amin < INF)

    ## a velocity profile \dot{s}(t)
    if (abs(delta) < EPSILON):
        vel_profile = Interpolate1D(0.0, 1.0, 0.0, 0.0, vmin, amin)
    else:
        vel_profile = InterpolateZeroVel1DWithDelta(0.0, 1.0, vmin, amin, delta)
    
    for vel_profile_ramp in vel_profile:
        dvect = x1vect - x0vect
        sdd = (vel_profile_ramp.a)*dvect
        sd_initial = (vel_profile_ramp.v)*dvect
        dur = vel_profile_ramp.T
        for j in range(ndof):
            ramp = Ramp(sd_initial[j], sdd[j], dur)
            rampslist = RampsList([ramp])
            rampslistnd0[j].Append(rampslist)
            
    rampslistnd = RampsListND(rampslistnd0)
    rampslistnd.SetInitialValue(x0vect)

    return rampslistnd
    
    
## Independent interpolation with arbitrary terminal velocities
def InterpolateArbitraryVelND(x0vect, x1vect, v0vect, v1vect, vmvect, amvect, delta = 0):
    ndof = len(x0vect)
    assert(ndof == len(x1vect))
    assert(ndof == len(v0vect))
    assert(ndof == len(v1vect))
    assert(ndof == len(vmvect))
    assert(ndof == len(amvect))

    rampslistnd0 = []
    Tmax = 0
    maxindex = 0
    for j in range(ndof):
        if (abs(delta) < EPSILON):
            rampslist = Interpolate1D(x0vect[j], x1vect[j],
                                      v0vect[j], v1vect[j],
                                      vmvect[j], amvect[j])
        else:
            rampslist = InterpolateArbitraryVel1DWithDelta(x0vect[j], x1vect[j],
                                                           v0vect[j], v1vect[j],
                                                           vmvect[j], amvect[j], delta)
        if (rampslist.isvoid):
            return RampsListND()
        rampslistnd0.append(rampslist)
        if (rampslist.duration > Tmax):
            Tmax = rampslist.duration
            maxindex = j
            
    rampslistnd = ReinterpolateND_FixedDuration(rampslistnd0, vmvect, amvect, 
                                                delta, maxindex)
    return rampslistnd


## Single DOF interpolation without considering a minimum-switch-time constraint
def Interpolate1D(x0, x1, v0, v1, vm, am):
    # print "Interpolate1D: (v0, v1) = ({0}, {1})".format(v0, v1)
    # print "             : (vm, am) = ({0}, {1})".format(vm, am)
    try:
        assert(abs(v0) <= vm + EPSILON)
    except AssertionError:
        print "Interpolate1D"
        print "v0 =", v0
        print "vm =", vm
        v0 = np.sign(v0)*vm
        
    try:
        assert(abs(v1) <= vm + EPSILON)
    except AssertionError:
        print "Interpolate1D"
        print "v1 =", v1
        print "vm =", vm
        v1 = np.sign(v1)*vm
    
    rampslist = InterpolatePP(x0, x1, v0, v1, am)
    
    if (len(rampslist) == 1):
        return rampslist
    
    rampslist = ReinterpolatePP2PLP(rampslist, vm)
    return rampslist


## Single DOF interpolation considering only a velocity bound
def InterpolatePP(x0, x1, v0, v1, am):
    d = x1 - x0
    dv = v1 - v0
    if (abs(d) < EPSILON) and (abs(dv) < EPSILON):
        ramp1 = Ramp(0.0, 0.0, 0.0, x0 = x0)
        ramp2 = Ramp(0.0, 0.0, 0.0)
        rampslist = RampsList([ramp1, ramp2])
        return rampslist
    
    ## d_straight is a distance covered if directly accel./decel. with max. magnitude
    ## from v0 to v1
    if (abs(dv) < EPSILON):
        d_straight = 0.0
    else:
        d_straight = (v1**2 - v0**2)/(2*np.sign(dv)*am)
        
    if (abs(d - d_straight) < EPSILON):
        sigma = 0
    else:
        sigma = np.sign(d - d_straight)
        
    if (sigma == 0):
        ## v1 can be reached by directly accel./decel. with max. magnitude from v0
        a = np.sign(dv)*am
        ramp1 = Ramp(v0, a, dv/a, x0 = x0)
        return RampsList([ramp1])

    sumvsq = v0**2 + v1**2
    vp = sigma*np.sqrt(0.5*sumvsq + sigma*am*d)
    a = sigma*am
    t0 = (vp - v0)/a
    t1 = (vp - v1)/a
    ramp1 = Ramp(v0, a, t0, x0 = x0)
    ramp2 = Ramp(vp, -a, t1)
    rampslist = RampsList([ramp1, ramp2])
    return rampslist


## Re-interpolation of a PP-trajectory to be a PLP-trajectory if the
## velocity bound is violated
def ReinterpolatePP2PLP(rampslist, vm):
    vp = rampslist[0].finalv
    if (abs(vp) <= vm + EPSILON):
        return rampslist

    h = abs(vp) - vm
    a0 = rampslist[0].a
    a1 = rampslist[1].a
    
    t0 = h/abs(a0)
    t1 = h/abs(a1)
    
    ramp1 = Ramp(rampslist[0].v, a0, rampslist[0].T - t0)
    ramp2 = Ramp(np.sign(vp)*vm, 0.0, t0 + t1 + 0.5*(t0 + t1)*h/vm)
    ramp3 = Ramp(np.sign(vp)*vm, a1, rampslist[1].T - t1)
    
    ramp1.x0 = rampslist.x0
    ramp2.x0 = ramp1.Eval(ramp1.T)
    ramp3.x0 = ramp2.Eval(ramp2.T)

    if (abs(ramp1.T) < EPSILON):
        return RampsList([ramp2, ramp3])
    elif (abs(ramp3.T) < EPSILON):
        return RampsList([ramp1, ramp2])
    else:
        return RampsList([ramp1, ramp2, ramp3])
        

## Single DOF interpolation with zero terminal velocities 
## considering a minimum-switch-time constraint
def InterpolateZeroVel1DWithDelta(x0, x1, vm, am, delta):
    rampslist = Interpolate1D(x0, x1, 0.0, 0.0, vm, am)
    
    if (len(rampslist) == 2):
        return FixSwitchTimeZeroVel_PP(rampslist, vm, am, delta)
    else:
        return FixSwitchTimeZeroVel_PLP(rampslist, vm, am, delta)


## Check and Fix a PP-trajectory with zero terminal velocities
## if a minimum-switch-time constraint is violated
def FixSwitchTimeZeroVel_PP(rampslist, vm, am, delta):
    if (rampslist[0].T >= delta):
        return rampslist

    vp = rampslist.distance/delta
    a0 = vp/delta
    ramp1 = Ramp(0.0, a0, delta, x0 = rampslist.x0)
    ramp2 = Ramp(vp, -a0, delta)
    rampslist = RampsList([ramp1, ramp2])
    return rampslist


## Check and Fix a PLP-trajectory with zero terminal velocities
## if a minimum-switch-time constraint is violated
def FixSwitchTimeZeroVel_PLP(rampslist, vm, am, delta):
    if (rampslist[0].T >= delta) and (rampslist[1].T >= delta):
        return rampslist
    
    d = rampslist.distance
    x0 = rampslist.x0
    
    if (am*delta <= vm):
        ## two-ramp
        vp = np.sign(d)*vm
        t0 = d/vp
        a0 = vp/t0
        ramp1 = Ramp(0.0, a0, t0, x0 = x0)
        ramp2 = Ramp(vp, -a0, t0)
        rampslistA = RampsList([ramp1, ramp2])
        
        ## three-ramp
        a0 = np.sign(d)*am
        vp = 0.5*(-a0*delta + np.sign(a0)*np.sqrt((a0*delta)**2 + 4*a0*d))
        if (vp/a0 < delta):
            ## (delta, delta, delta)
            vp = 0.5*d/delta
            ramp1 = Ramp(0.0, vp/delta, delta, x0 = x0)
            ramp2 = Ramp(vp, 0.0, delta)
            ramp3 = Ramp(vp, -vp/delta, delta)
        else:
            ## (> delta, delta, > delta)
            ramp1 = Ramp(0.0, a0, vp/a0, x0 = x0)
            ramp2 = Ramp(vp, 0.0, delta)
            ramp3 = Ramp(vp, -a0, vp/a0)
        rampslistB = RampsList([ramp1, ramp2, ramp3])

        if (rampslistA.duration <= rampslistB.duration):
            return rampslistA
        else:
            return rampslistB
    else:
        if (abs(d) <= vm*delta):
            ## two-ramp (delta, delta)
            vp = d/delta
            a0 = vp/delta
            ramp1 = Ramp(0.0, a0, delta, x0 = x0)
            ramp2 = Ramp(vp, -a0, delta)
            return RampsList([ramp1, ramp2])
        elif (abs(d) > 2*vm*delta):
            ## (delta, > delta, delta)
            vp = np.sign(d)*vm
            a0 = vp/delta
            ramp1 = Ramp(0.0, a0, delta, x0 = x0)
            ramp3 = Ramp(vp, -a0, delta)
            d_rem = d - (ramp1.distance + ramp3.distance)
            ramp2 = Ramp(vp, 0.0, d_rem/vp)
            return RampsList([ramp1, ramp2, ramp3])
        elif (abs(d) <= 1.5*vm*delta):
            ## two-ramp (>= delta, >= delta)
            vp = np.sign(d)*vm
            t = d/vp
            a0 = vp/t
            ramp1 = Ramp(0.0, a0, t, x0 = x0)
            ramp2 = Ramp(vp, -a0, t)
            return RampsList([ramp1, ramp2])
        else:
            ## three-ramp (delta, delta, delta)
            vp = 0.5*d/delta
            a0 = vp/delta
            ramp1 = Ramp(0.0, a0, delta, x0 = x0)
            ramp2 = Ramp(vp, 0.0, delta)
            ramp3 = Ramp(vp, -a0, delta)
            return RampsList([ramp1, ramp2, ramp3])


## Single DOF interpolation with arbitrary terminal velocities
## considering a minimum-switch-time constraint
def InterpolateArbitraryVel1DWithDelta(x0, x1, v0, v1, vm, am, delta):
    rampslist = Interpolate1D(x0, x1, v0, v1, vm, am)
    
    if (len(rampslist) == 1):
        return FixSwitchTimeArbitraryVel_OneRamp(rampslist, vm, am, delta)
    elif (len(rampslist) == 2):
        return FixSwitchTimeArbitraryVel_PP(rampslist, vm, am, delta)
    else:
        return FixSwitchTimeArbitraryVel_PLP(rampslist, vm, am, delta)


## Check and Fix a one-ramp trajectory if a minimum-switch-time constraint is violated
## The case of one-ramp trajectory is likely to occur in shortcutting process only.
def FixSwitchTimeArbitraryVel_OneRamp(rampslist, vm, am, delta):
    if (rampslist.duration >= delta):
        return rampslist

    CASE = "ONE-RAMP"
    
    v0 = rampslist[0].v
    v1 = rampslist[0].finalv
    x0 = rampslist[0].x0
    d = rampslist.distance    
    
    if (np.sign(v0*v1) > 0):
        ## v0 and v1 are both positive (or negative)
        a0new = -np.sign(v0)*am
        vpnew = -np.sign(v0)*min(abs(v0), abs(v1))
        t0new = (abs(v0) + abs(vpnew))/am
        t1new = (abs(vpnew) + abs(v1))/am
        if (t0new >= delta) and (t1new >= delta):
            ramp1A = Ramp(v0, a0new, t0new, x0 = x0)
            ramp2A = Ramp(vpnew, -a0new, t1new)
            rampslistA = RampsList([ramp1A, ramp2A])
            if PRINTCASE:
                print CASE + " --> PP3C"
            return rampslistA
        else:
            if (t0new < t1new):
                ## modify according to PP1
                rampslistB = PP_1(rampslist, vm, am, delta)
                if PRINTCASE:
                    print CASE + " --> PP1"
                return rampslistB
            else:
                ## modify according to PP2
                rampslistC = PP_2(rampslist, vm, am, delta)
                if PRINTCASE:
                    print CASE + " --> PP2 "
                return rampslistC
    elif (np.sign(v0*v1) < 0):
        anew = (v1 - v0)/delta
        if (abs(anew) < am + EPSILON):
            ramp1D = Ramp(v0, anew, delta, x0 = x0)
            rampslistD = RampsList([ramp1D])
            if PRINTCASE:
                print CASE + " --> ONE-RAMP "
            return rampslistD
        else:
            ## no solution available
            return RampsList()
    else:
        ## fix with two delta-ramps
        vpnew = d/delta - 0.5*(v0 + v1)
        a0new = (vpnew - v0)/delta
        a1new = (v1 - vpnew)/delta
        ramp1E = Ramp(v0, a0new, delta, x0 = x0)
        ramp2E = Ramp(vpnew, a1new, delta)
        rampslistE = RampsList([ramp1E, ramp2E])

        if PRINTCASE:
            print CASE + " --> TWO-DELTA-RAMP"
        return rampslistE
        

## Check and Fix a PP-trajectory with arbitrary terminal velocities
## if a minimum-switch-time constraint is violated
def FixSwitchTimeArbitraryVel_PP(rampslist, vm, am, delta):
    assert(len(rampslist) == 2)

    t0 = rampslist[0].T
    t1 = rampslist[1].T
    
    if (t0 < delta) and (t1 >= delta):
        ## PP1
        return PP_1(rampslist, vm, am, delta)
    elif (t0 >= delta) and (t1 < delta):
        ## PP2
        return PP_2(rampslist, vm, am, delta)
    elif (t0 < delta) and (t1 < delta):
        ## PP3
        return PP_3(rampslist, vm, am, delta)
    else:
        # no fixing needed
        return rampslist


## Check and Fix a PLP-trajectory with arbitrary terminal velocities
## if a minimum-switch-time constraint is violated
def FixSwitchTimeArbitraryVel_PLP(rampslist, vm, am, delta):
    assert(len(rampslist) == 3)

    t0 = rampslist[0].T
    t1 = rampslist[1].T
    t2 = rampslist[2].T

    if (t0 < delta) and (t1 >= delta) and (t2 >= delta):
        ## PLP1
        return PLP_1(rampslist, vm, am, delta)
    elif (t0 >= delta) and (t1 >= delta) and (t2 < delta):
        ## PLP2
        return PLP_2(rampslist, vm, am, delta)
    elif (t0 >= delta) and (t1 < delta) and (t2 >= delta):
        ## PLP3
        return PLP_3(rampslist, vm, am, delta)
    elif (t0 < delta) and (t1 < delta) and (t2 >= delta):
        ## PLP4
        return PLP_4(rampslist, vm, am, delta)
    elif (t0 >= delta) and (t1 < delta) and (t2 < delta):
        ## PLP5
        return PLP_5(rampslist, vm, am, delta)
    elif (t0 < delta) and (t1 >= delta) and (t2 < delta):
        ## PLP6
        return PLP_6(rampslist, vm, am, delta)
    elif (t0 < delta) and (t1 < delta) and (t2 < delta):
        ## PLP7
        return PLP_7(rampslist, vm, am, delta)
    else:
        # no fixing needed
        return rampslist


def PP_1(rampslist, vm, am, delta):
    """
    First, stretch the duration of the first ramp to \delta
    A
    - no further correction is needed
    B
    - re-interpolating with 2 delta-ramps is better
    C
    - re-interpolating with one ramp is better
    """
    firstramp = rampslist[0]
    lastramp = rampslist[-1]
    t0 = firstramp.T
    t1 = lastramp.T
    v0 = firstramp.v
    v1 = rampslist.Evald(rampslist.duration)
    a0 = firstramp.a
    a1 = lastramp.a
    d = rampslist.distance
    x0 = firstramp.x0
    vp = lastramp.v
    
    CASEA = "PP1A"
    CASEB = "PP1B"
    CASEC = "PP1C"
    
    if (abs(a1) < EPSILON):
        ## the longer ramp has zero acceleration
        ## have to have a separate operation since the normal procedure requires
        ## dividing by a1
        
        ## stretch the first ramp
        ramp1A = Ramp(v0, (vp - v0)/delta, delta, x0 = x0)
        
        d_rem = d - ramp1A.distance
        t1new = d_rem/vp
        
        if (t1new >= delta):
            ## no further correction is needed
            ramp2A = Ramp(vp, 0.0, t1new)
            rampslistA = RampsList([ramp1A, ramp2A])
            if PRINTCASE:
                print CASEA + "(a1 == 0)"
            return rampslistA

        else:
            ## otherwise, make a comparison between 2-delta-ramp & one-ramp
            vpnew = d/delta - 0.5*(v0 + v1)
            a0new = (vpnew - v0)/delta
            a1new = (v1 - vpnew)/delta
            
            if (abs(a0new) <= am) and (abs(a1new) <= am) and (abs(vpnew) <= vm):
                ## 2-delta-ramp
                ramp1B = Ramp(v0, a0new, delta, x0 = x0)
                ramp2B = Ramp(vpnew, a1new, delta)
                rampslistB = RampsList([ramp1B, ramp2B])
            else:
                rampslistB = RampsList()

            ## one-ramp
            if (abs(v0 + v1) < EPSILON):
                rampslistC = RampsList()
            else:
                tnew = (2*d)/(v0 + v1)
                anew = (v1 - v0)/tnew
                if (abs(anew) > am + EPSILON) or (tnew < delta):
                    rampslistC = RampsList()
                else:
                    ramp1C = Ramp(v0, anew, tnew, x0 = x0)
                    rampslistC = RampsList([ramp1C])

            if (not (rampslistB.isvoid)) or (not (rampslistC.isvoid)):                
                if (rampslistB.isvoid):
                    if PRINTCASE:
                        print CASEC + "(a1 == 0)"
                    return rampslistC
                elif (rampslistC.isvoid):
                    if PRINTCASE:
                        print CASEB + "(a1 == 0)"
                    return rampslistB
                elif (rampslistB.duration <= rampslistC.duration):
                    if PRINTCASE:
                        print CASEB + "(a1 == 0)"
                    return rampslistB
                else:
                    if PRINTCASE:
                        print CASEC + "(a1 == 0)"
                    return rampslistC
            else:
                ## this cannot happen
                print "[PP1::warning] no solution"
                raw_input()
                return RampsList()
    ## end special case (a1 == 0)

    k = a1*delta
    vpnew = 0.5*(k - np.sign(a1)*np.sqrt(k**2 + 4*(k*v0 + v1**2) - 8*a1*d))
    t1new = (v1 - vpnew)/a1
    if (t1new < delta):
        ## 2-delta-ramp
        vpnew = d/delta - 0.5*(v0 + v1)
        a0new = (vpnew - v0)/delta
        a1new = (v1 - vpnew)/delta
        
        if (abs(a0new) <= am) and (abs(a1new) <= am) and (abs(vpnew) <= vm):
            ## 2-delta-ramp
            ramp1B = Ramp(v0, a0new, delta, x0 = x0)
            ramp2B = Ramp(vpnew, a1new, delta)
            rampslistB = RampsList([ramp1B, ramp2B])
        else:
            rampslistB = RampsList()
            
        ## one-ramp
        if (abs(v0 + v1) < EPSILON):
            rampslistC = RampsList()
        else:
            tnew = (2*d)/(v0 + v1)
            anew = (v1 - v0)/tnew
            if (abs(anew) > am + EPSILON) or (tnew < delta):
                rampslistC = RampsList()
            else:
                ramp1C = Ramp(v0, anew, tnew, x0 = x0)
                rampslistC = RampsList([ramp1C])

        if (not (rampslistB.isvoid)) or (not (rampslistC.isvoid)):                
            if (rampslistB.isvoid):
                if PRINTCASE:
                    print CASEC
                return rampslistC
            elif (rampslistC.isvoid):
                if PRINTCASE:
                    print CASEB
                return rampslistB
            elif (rampslistB.duration <= rampslistC.duration):
                if PRINTCASE:
                    print CASEB
                return rampslistB
            else:
                if PRINTCASE:
                    print CASEC
                return rampslistC
        else:
            ## this cannot happen
            print "[PP1::warning] no solution"
            raw_input()
            return RampsList()
    else:
        ## no further correction is needed
        a0new = (vpnew - v0)/delta
        ramp1A = Ramp(v0, a0new, delta, x0 = x0)
        ramp2A = Ramp(vpnew, a1, t1new)
        rampslistA = RampsList([ramp1A, ramp2A])
        if PRINTCASE:
            print CASEA
        return rampslistA
    

def PP_2(rampslist, vm, am, delta):
    """
    First, stretch the duration of the second ramp to \delta
    A
    - no further correction is needed
    B
    - re-interpolating with 2 delta-ramps is better
    C
    - re-interpolating with one ramp is better
    """
    firstramp = rampslist[0]
    lastramp = rampslist[-1]
    t0 = firstramp.T
    t1 = lastramp.T
    v0 = firstramp.v
    v1 = rampslist.Evald(rampslist.duration)
    a0 = firstramp.a
    a1 = lastramp.a
    d = rampslist.distance
    x0 = firstramp.x0
    vp = lastramp.v
    
    CASEA = "PP2A"
    CASEB = "PP2B"
    CASEC = "PP2C"
    
    if (abs(a0) < EPSILON):
        ## the longer ramp has zero acceleration
        ## have to have a separate operation since the normal procedure requires
        ## dividing by a0

        ## stretch the last ramp
        ramp2A = Ramp(vp, (v1 - vp)/delta, delta)

        d_rem = d - ramp2A.distance
        t0new = d_rem/vp
        
        if (t0new >= delta):
            ## no further correction is needed
            ramp1A = Ramp(v0, 0.0, t0new, x0 = x0)
            rampslistA = RampsList([ramp1A, ramp2A])
            if PRINTCASE:
                print CASEA + "(a0 == 0)"
            return rampslistA
        else:
            ## otherwise, make a comparison between 2-delta-ramp & one-ramp
            vpnew = d/delta - 0.5*(v0 + v1)
            a0new = (vpnew - v0)/delta
            a1new = (v1 - vpnew)/delta
            
            if (abs(a0new) <= am) and (abs(a1new) <= am) and (abs(vpnew) <= vm):
                ## 2-delta-ramp
                ramp1B = Ramp(v0, a0new, delta, x0 = x0)
                ramp2B = Ramp(vpnew, a1new, delta)
                rampslistB = RampsList([ramp1B, ramp2B])
            else:
                rampslistB = RampsList()

            ## one-ramp
            if (abs(v0 + v1) < EPSILON):
                rampslistC = RampsList()
            else:
                tnew = (2*d)/(v0 + v1)
                anew = (v1 - v0)/tnew
                if (abs(anew) > am + EPSILON) or (tnew < delta):
                    rampslistC = RampsList()
                else:
                    ramp1C = Ramp(v0, anew, tnew, x0 = x0)
                    rampslistC = RampsList([ramp1C])
               
            if (not (rampslistB.isvoid)) or (not (rampslistC.isvoid)):                
                if (rampslistB.isvoid):
                    if PRINTCASE:
                        print CASEC + "(a0 == 0)"
                    return rampslistC
                elif (rampslistC.isvoid):
                    if PRINTCASE:
                        print CASEB + "(a0 == 0)"
                    return rampslistB
                elif (rampslistB.duration <= rampslistC.duration):
                    if PRINTCASE:
                        print CASEB + "(a0 == 0)"
                    return rampslistB
                else:
                    if PRINTCASE:
                        print CASEC + "(a0 == 0)"
                    return rampslistC
            else:
                ## this cannot happen
                print "[PP2::warning] no solution"
                raw_input()
                return RampsList()
    ## end special case (a0 == 0)
                
    k = a0*delta
    vpnew = 0.5*( - k + np.sign(a0)*np.sqrt(k**2 - 4*(k*v1 - v0**2) + 8*a0*d))
    t0new = (vpnew - v0)/a0
    if (t0new < delta):
        ## 2-delta-ramp
        vpnew = d/delta - 0.5*(v0 + v1)
        a0new = (vpnew - v0)/delta
        a1new = (v1 - vpnew)/delta
        ramp1B = Ramp(v0, a0new, delta, x0 = x0)
        ramp2B = Ramp(vpnew, a1new, delta)
        rampslistB = RampsList([ramp1B, ramp2B])

        ## one-ramp
        if (abs(v0 + v1) < EPSILON):
            rampslistC = RampsList()
        else:
            tnew = (2*d)/(v0 + v1)
            anew = (v1 - v0)/tnew
            if (abs(anew) > am + EPSILON) or (tnew < delta):
                rampslistC = RampsList()
            else:
                ramp1C = Ramp(v0, anew, tnew, x0 = x0)
                rampslistC = RampsList([ramp1C])

        if (not (rampslistB.isvoid)) or (not (rampslistC.isvoid)):                
            if (rampslistB.isvoid):
                if PRINTCASE:
                    print CASEC
                return rampslistC
            elif (rampslistC.isvoid):
                if PRINTCASE:
                    print CASEB
                return rampslistB
            elif (rampslistB.duration <= rampslistC.duration):
                if PRINTCASE:
                    print CASEB
                return rampslistB
            else:
                if PRINTCASE:
                    print CASEC
                return rampslistC
        else:
            ## this cannot happen
            print "[PP1::warning] no solution"
            raw_input()
            return RampsList()
    else:
        ## no further correction is needed
        a1new = (v1 - vpnew)/delta
        ramp1A = Ramp(v0, a0, t0new, x0 = x0)
        ramp2A = Ramp(vpnew, a1new, delta)
        rampslistA = RampsList([ramp1A, ramp2A])
        if PRINTCASE:
            print CASEA
        return rampslistA


def PP_3(rampslist, vm, am, delta):
    """
    A
    - re-interpolation with 2 delta-ramps is better
    B
    - re-interpolation with a straight line is better
    
    In case, A & B are both invalid, try flipping
    """
    firstramp = rampslist[0]
    lastramp = rampslist[-1]
    t0 = firstramp.T
    t1 = lastramp.T
    v0 = firstramp.v
    v1 = rampslist.Evald(rampslist.duration)
    a0 = firstramp.a
    a1 = lastramp.a
    d = rampslist.distance
    x0 = firstramp.x0
    vp = lastramp.v
    
    CASEA = "PP3A"
    CASEB = "PP3B"
    CASEC = "PP3C"
    
    ## try PP3A
    vpnew = d/delta - 0.5*(v0 + v1)
    a0new = (vpnew - v0)/delta
    a1new = (v1 - vpnew)/delta
    if (abs(a0new) > am) or (abs(a1new) > am) or (abs(vpnew) > vm):
        rampslistA = RampsList()
    else:
        ramp1A = Ramp(v0, a0new, delta, x0 = x0)
        ramp2A = Ramp(vpnew, a1new, delta)
        rampslistA = RampsList([ramp1A, ramp2A])
    
    ## try PP3B
    if (abs(v0 + v1) < EPSILON):
        rampslistB = RampsList()
    else:
        tnew = (2*d)/(v0 + v1)
        anew = (v1 - v0)/tnew
        if (tnew < delta) or (abs(anew) > am):
            rampslistB = RampsList()
        else:
            ramp1B = Ramp(v0, anew, tnew, x0 = x0)
            rampslistB = RampsList([ramp1B])
        
    if (not (rampslistA.isvoid)) or (not (rampslistB.isvoid)):
        if (rampslistA.isvoid):
            if PRINTCASE:
                print CASEB
            return rampslistB
        elif (rampslistB.isvoid):
            if PRINTCASE:
                print CASEA
            return rampslistA
        else:
            if (rampslistB.duration <= rampslistA.duration):
                if PRINTCASE:
                    print CASEB
                return rampslistB
            else:
                if PRINTCASE:
                    print CASEA
                return rampslistA
    ## now PP3A & PP3B are both invalid
    ## try PP3C
    ## first, see if flipping with both acclerations saturated is eligible
    if (abs(a0) < EPSILON):
        a0new = a1
        a1new = -a1
    elif (abs(a1) < EPSILON):
        a0new = -a0
        a1new = -a0new
    else:
        a0new = -a0
        a1new = -a1
    vpsq = 0.5*(v0*v0 + v1*v1) + a0new*d
    if (vpsq >= 0) and (abs(a0new) > EPSILON) and (abs(a1new) > EPSILON):
        ## both ramps can saturate the acceleration bound at the same time
        vpnew = np.sqrt(vpsq)
        t0new = (vpnew - v0)/a0new
        t1new = (v1 - vpnew)/a1new
        if (t0new >= delta) and (t1new >= delta):
            ramp1C = Ramp(v0, a0new, t0new, x0 = x0)
            ramp2C = Ramp(vpnew, a1new, t1new)
            rampslistC = RampsList([ramp1C, ramp2C])
            if PRINTCASE:
                print CASEC
            return rampslistC
        else:
            vpnew = -vpnew
            t0new = (vpnew - v0)/a0new
            t1new = (v1 - vpnew)/a1new
            if (t0new >= delta) and (t1new >= delta):
                ramp1C = Ramp(v0, a0new, t0new, x0 = x0)
                ramp2C = Ramp(vpnew, a1new, t1new)
                rampslistC = RampsList([ramp1C, ramp2C])
                if PRINTCASE:
                    print CASEC
                return rampslistC
    ## now we have that vpsq < 0
    if (t0 >= t1):
        ## t0new <= t1new
        ## modify according to PP1
        k = a1new*delta
        discriminant = k**2 + 4*(k*v0 + v1**2) - 8*a1new*d
        if (discriminant < 0):
            ## to be removed
            # print "ENCOUNTER A NEGATIVE NUMBER IN THE SQUARE ROOT"
            # print "vmax = {0}\namax = {1}".format(vm, am)
            # print rampslist
            # raw_input()
            return RampsList()
            
        vpnew = 0.5*(k - np.sign(a1new)*np.sqrt(discriminant))
        t1new = (v1 - vpnew)/a1new
        if (t1new < delta):
            ## 2-delta-ramp
            vpnew = d/delta - 0.5*(v0 + v1)
            a0new = (vpnew - v0)/delta
            a1new = (v1 - vpnew)/delta
            if (abs(a0new) > am) or (abs(a1new) > am) or (abs(vpnew) > vm):
                rampslistB = RampsList()
            else:
                ramp1B = Ramp(v0, a0new, delta, x0 = x0)
                ramp2B = Ramp(vpnew, a1new, delta)
                rampslistB = RampsList([ramp1B, ramp2B])                
            ## one-ramp
            if (abs(v0 + v1) < EPSILON):
                rampslistC = RampsList()
            else:
                tnew = (2*d)/(v0 + v1)
                anew = (v1 - v0)/tnew
                if (abs(anew) > am) or (tnew < delta):
                    rampslistC = RampsList()
                else:
                    ramp1C = Ramp(v0, anew, tnew, x0 = x0)
                    rampslistC = RampsList([ramp1C])
            
            if (not (rampslistC.isvoid)) and (rampslistB.duration > rampslistC.duration):
                if PRINTCASE:
                    print CASEC + " --> PP1C"
                return rampslistC
            else:
                if PRINTCASE:
                    print CASEC + " --> PP1B"
                return rampslistB
        else:
            ## no further correctin is needed
            a0new = (vpnew - v0)/delta
            ramp1A = Ramp(v0, a0new, delta, x0 = x0)
            ramp2A = Ramp(vpnew, a1new, t1new)
            rampslistA = RampsList([ramp1A, ramp2A])
            if PRINTCASE:
                print CASEC + " --> PP1A"
            return rampslistA
    else:
        ## t0new > t1new
        ## modify according to PP2
        k = a0new*delta
        discriminant = k**2 - 4*(k*v1 - v0**2) + 8*a0new*d
        if (discriminant < 0):
            ## to be removed
            # print "ENCOUNTER A NEGATIVE NUMBER IN THE SQUARE ROOT"
            # print "vmax = {0}\namax = {1}".format(vm, am)
            # print rampslist
            # raw_input()
            return RampsList()
            
        vpnew = 0.5*( - k + np.sign(a0new)*np.sqrt(discriminant))
        t0new = (vpnew - v0)/a0new
        if (t0new < delta):
            ## 2-delta-ramp
            vpnew = d/delta - 0.5*(v0 + v1)
            a0new = (vpnew - v0)/delta
            a1new = (v1 - vpnew)/delta
            if (abs(a0new) > am) or (abs(a1new) > am) or (abs(vpnew) > vm):
                rampslistB = RampsList()
            else:
                ramp1B = Ramp(v0, a0new, delta, x0 = x0)
                ramp2B = Ramp(vpnew, a1new, delta)
                rampslistB = RampsList([ramp1B, ramp2B])
            
            ## one-ramp
            if (abs(v0 + v1) < EPSILON):
                rampslistC = RampsList()
            else:
                tnew = (2*d)/(v0 + v1)
                anew = (v1 - v0)/tnew
                if (abs(anew) > am) or (tnew < delta):
                    rampslistC = RampsList()
                else:
                    ramp1C = Ramp(v0, anew, tnew, x0 = x0)
                    rampslistC = RampsList([ramp1C])
                
            if (not (rampslistC.isvoid)) and (rampslistB.duration > rampslistC.duration):
                if PRINTCASE:
                    print CASEC + " --> PP1C"
                return rampslistC
            else:
                if PRINTCASE:
                    print CASEC + " --> PP1B"
                return rampslistB
        else:
            ## no further correctin is needed
            a1new = (v1 - vpnew)/delta
            ramp1A = Ramp(v0, a0new, t0new, x0 = x0)
            ramp2A = Ramp(vpnew, a1new, delta)
            rampslistA = RampsList([ramp1A, ramp2A])
            if PRINTCASE:
                print CASEC + " --> PP2A"
            return rampslistA


def PLP_1(rampslist, vm, am, delta):
    firstramp = copy.deepcopy(rampslist[0])
    middleramp = copy.deepcopy(rampslist[1])
    lastramp = copy.deepcopy(rampslist[2])

    t0 = firstramp.T
    t1 = middleramp.T
    t2 = lastramp.T
    
    v0 = firstramp.v
    v1 = lastramp.finalv
    
    CASEA = "PLP1A"
    CASEB = "PLP1B"
    
    x0 = firstramp.x0
    d = rampslist.distance
    d0 = firstramp.distance + middleramp.distance
    vp = middleramp.v

    """
    A 
    - stretch the duration of the first ramp to \delta
    - fix the remaining two with InterpolateArbitraryVel1DWithDelta
    """
    a0new = (vp - v0)/delta
    ramp1A = Ramp(v0, a0new, delta, x0 = x0)
    d_rem = d - ramp1A.distance ## remaining distance after stretching
    lasttworamps = InterpolateArbitraryVel1DWithDelta(0, d_rem, vp, v1, vm, am, delta)
    if (len(lasttworamps) == 1):
        rampslistA = RampsList([ramp1A, lasttworamps[0]])
    elif (len(lasttworamps) == 2):
        rampslistA = RampsList([ramp1A, lasttworamps[0], lasttworamps[1]])

    """
    B
    - merge the first two ramps into one ramp
    - the last ramp remains the same (the new second ramp is the original third ramp)
    """
    if (abs(vp + v0) < EPSILON):
        rampslistB = RampsList()
    else:
        t0new = 2*d0/(vp + v0)
        a0new = (vp - v0)/t0new
        ramp1B = Ramp(v0, a0new, t0new, x0 = x0)
        rampslistB = RampsList([ramp1B, lastramp])
    
    if (not (rampslistA.isvoid)) or (not (rampslistB.isvoid)):
        if (rampslistA.isvoid):
            if PRINTCASE:
                print CASEB
            return rampslistB
        elif (rampslistB.isvoid):
            if PRINTCASE:
                print CASEA
            return rampslistA
        else:
            if (rampslistB.duration <= rampslistA.duration):
                if PRINTCASE:
                    print CASEB
                return rampslistB
            else:
                if PRINTCASE:
                    print CASEA
                return rampslistA
    else:
        ## this cannot happen
        print "[PLP1::warning] no solution"
        raw_input()
        return RampsList()


def PLP_2(rampslist, vm, am, delta):
    firstramp = copy.deepcopy(rampslist[0])
    middleramp = copy.deepcopy(rampslist[1])
    lastramp = copy.deepcopy(rampslist[2])

    t0 = firstramp.T
    t1 = middleramp.T
    t2 = lastramp.T
    
    v0 = firstramp.v
    v1 = lastramp.finalv
    
    CASEA = "PLP2A"
    CASEB = "PLP2B"
    
    x0 = firstramp.x0
    d = rampslist.distance
    d1 = middleramp.distance + lastramp.distance
    vp = middleramp.v

    """
    A 
    - stretch the duration of the last ramp to \delta
    - fix the remaining two with InterpolateArbitraryVel1DWithDelta
    """
    a2new = (v1 - vp)/delta
    ramp3A = Ramp(vp, a2new, delta)
    d_rem = d - ramp3A.distance ## remaining distance after stretching
    firsttwo = InterpolateArbitraryVel1DWithDelta(0, d_rem, v0, vp, vm, am, delta)
    firsttwo[0].x0 = x0
    if (len(firsttwo) == 1):
        rampslistA = RampsList([firsttwo[0], ramp3A])
    elif( len(firsttwo)== 2):
        rampslistA = RampsList([firsttwo[0], firsttwo[1], ramp3A])

    """
    B
    - merge the last two ramps into one ramp
    - the first ramp remains the same
    """
    if (abs(vp + v1) < EPSILON):
        rampslistB = RampsList()
    else:
        t1new = 2*d1/(vp + v1)
        a1new = (v1 - vp)/t1new
        ramp2B = Ramp(vp, a1new, t1new)
        rampslistB = RampsList([firstramp, ramp2B])
    
    if (not (rampslistA.isvoid)) or (not (rampslistB.isvoid)):
        if (rampslistA.isvoid):
            if PRINTCASE:
                print CASEB
            return rampslistB
        elif (rampslistB.isvoid):
            if PRINTCASE:
                print CASEA
            return rampslistA
        else:
            if (rampslistB.duration <= rampslistA.duration):
                if PRINTCASE:
                    print CASEB
                return rampslistB
            else:
                if PRINTCASE:
                    print CASEA
                return rampslistA
    else:
        ## this cannot happen
        print "[PLP2::warning] no solution"
        raw_input()
        return RampsList()


def PLP_3(rampslist, vm, am, delta):
    """
    PLP 3A
    """
    if rampslist[0].a > 0:
        am_ = am
        vm_ = vm
    else:
        am_ = -1.0*am
        vm_ = -1.0*vm
    v0 = rampslist[0].v
    v1 = rampslist[-1].finalv
    d = rampslist.distance
    k = 0.5*delta + v0/am_
    h = 0.5*delta + v1/am_
    r2 = (1./(am_**2)) * (v0*v0 + v1*v1 + 2*d*am_ + 0.5*am_*am_*delta*delta)

    t0l = max(delta, (v1 - v0)/am_)
    t0u = (vm_ - v0)/am_
    it0_1 = Interval(t0l, t0u)
    t2l = -h + np.sqrt(r2 - (k + t0u)**2)
    t2u = -h + np.sqrt(r2 - (k + t0l)**2)
    if t2l > t2u:
        print 'sign(h)', np.sign(h)
        t2l = -h - np.sqrt(r2 - (k + t0u)**2)
        t2u = -h - np.sqrt(r2 - (k + t0l)**2)

    it2_1 = Interval(t2l, t2u)
    
    t2l = max(delta, (v0 - v1)/am_)
    t2u = (vm_ - v1)/am_
    it2_2 = Interval(t2l, t2u)
    t0l = -k + np.sqrt(r2 - (h + t2u)**2)
    t0u = -k + np.sqrt(r2 - (h + t2l)**2)
    if t0l > t0u:
        print 'sign(k)', np.sign(k)
        t0l = -k - np.sqrt(r2 - (h + t2u)**2)
        t0u = -k - np.sqrt(r2 - (h + t2l)**2)

    it0_2 = Interval(t0l, t0u)
    
    it0 = Intersect(it0_1, it0_2)
    it2 = Intersect(it2_1, it2_2)

    if not (it0.isvoid):
        dura = it0.l + delta + it2.u
        durb = it0.u + delta + it2.l
        if dura <= durb:
            t0 = it0.l
            t2 = it2.u
            vp0 = v0 + am_*t0
            vp1 = v1 + am_*t2
        else:
            t0 = it0.u
            t2 = it2.l
            vp0 = v0 + am_*t0
            vp1 = v1 + am_*t2
        ramp1 = Ramp(v0, am_, t0, x0=rampslist[0].x0)
        ramp2 = Ramp(vp0, (vp1 - vp0)/delta, delta)
        ramp3 = Ramp(vp1, -am_, t2)
        rampslistA = RampsList([ramp1, ramp2, ramp3])
    else:
        rampslistA = RampsList()

    """
    PLP 3B
    """
    firstramp = copy.deepcopy(rampslist[0])
    middleramp = copy.deepcopy(rampslist[1])
    lastramp = copy.deepcopy(rampslist[2])

    t0 = firstramp.T
    t1 = middleramp.T
    t2 = lastramp.T

    v0 = firstramp.v
    v1 = lastramp.finalv
    
    x0 = firstramp.x0
    d = rampslist.distance
    d0 = firstramp.distance + middleramp.distance
    d1 = middleramp.distance + lastramp.distance
    vp = middleramp.v

    
    lasttworamps = InterpolateArbitraryVel1DWithDelta(0, d1, vp, v1, vm, am, delta)
    if (len(lasttworamps) == 2):
        rampslistB1 = RampsList([firstramp, lasttworamps[0], lasttworamps[1]])
    else:
        rampslistB1 = RampsList([firstramp, lasttworamps[0]])

    firsttworamps = InterpolateArbitraryVel1DWithDelta(0, d0, v0, vp, vm, am, delta)
    firsttworamps[0].x0 = x0
    if (len(firsttworamps) == 2):
        rampslistB2 = RampsList([firsttworamps[0], firsttworamps[1], lastramp])
    else:
        rampslistB2 = RampsList([firsttworamps[0], lastramp])
    # compare both cases
    if (rampslistB1.duration < rampslistB2.duration):
        rampslistB = rampslistB1
    else:
        rampslistB = rampslistB2

    """
    PLP 3C (3-delta)
    """
    sum_vp = d/delta - 0.5*(v0 + v1)
    sumvprange = Interval(v0 + v1 - 2*am*delta, v0 + v1 + 2*am*delta)
    if not sumvprange.contain(sum_vp):
        rampslistC = RampsList()
    else:
        vp0range = Interval(v0 - am*delta, v0 + am*delta)

        vp1range1 = Interval(sum_vp - vp0range.u, sum_vp - vp0range.l)
        vp1range2 = Interval(v1 - am*delta, v1 + am*delta)
        vp1range = Intersect(vp1range1, vp1range2)
        if vp1range.isvoid:
            rampslistC = RampsList()
        else:
            passed = False
            while not passed:
                vp1 = RNG.uniform(vp1range.l, vp1range.u)
                vp0 = sum_vp - vp1
                passed = abs((vp1 - vp0)/delta) <= am
            ramp1 = Ramp(v0, (vp0 - v0)/delta, delta, x0=rampslist[0].x0)
            ramp2 = Ramp(vp0, (vp1 - vp0)/delta, delta)
            ramp3 = Ramp(vp1, (v1 - vp1)/delta, delta)
            rampslistC = RampsList([ramp1, ramp2, ramp3])
            
    if not rampslistC.isvoid:
        if not rampslistB.isvoid:
            if rampslistB.duration <= rampslistC.duration:
                return rampslistB
            else:
                return rampslistC
        else:
            return rampslistC
        
    if rampslistA.isvoid:
        if len(rampslistB) == 3:
            raw_input('ERROR')
        else:
            print 'TWO RAMPS'
            return rampslistB
    else:
        if rampslistB.isvoid:
            return rampslistA
        else:
            if rampslistB.duration < rampslistA.duration - 1e-8:
                if len(rampslistB) == 3:
                    raw_input('ERROR')
                else:
                    print 'TWO RAMPS'
                    return rampslistB
            else:
                return rampslistA


def PLP_4(rampslist, vm, am, delta):
    firstramp = copy.deepcopy(rampslist[0])
    middleramp = copy.deepcopy(rampslist[1])
    lastramp = copy.deepcopy(rampslist[2])

    t0 = firstramp.T
    t1 = middleramp.T
    t2 = lastramp.T
    
    v0 = firstramp.v
    v1 = lastramp.Evald(t2)
    
    CASEA = "PLP4A"
    CASEB = "PLP4B"
    CASEC = "PLP4C"
    
    x0 = firstramp.x0
    d = rampslist.distance
    vp = middleramp.v
    
    """
    A
    - stretch the first ramp
    - fix the remaining distance with InterpolateArbitraryVel1DWithDelta
    """
    a0new = (vp - v0)/delta
    ramp1A = Ramp(v0, a0new, delta, x0 = x0)
    d_rem = d - ramp1A.distance
    res = InterpolateArbitraryVel1DWithDelta(0, d_rem, vp, v1, vm, am, delta)
    if (res.isvoid):
        rampslistA = RampsList()
    elif (len(res) == 1):
        rampslistA = RampsList([ramp1A, res[0]])
    elif (len(res) == 2):
        rampslistA = RampsList([ramp1A, res[0], res[1]])

    """
    B
    - merge the first two ramps together
    - the original last ramp remains the same
    """
    d_rem = d - lastramp.distance
    if (abs(v0 + vp) < EPSILON):
        rampslistB = RampsList()
    else:
        t0new = (2*d_rem)/(v0 + vp)
        if (t0new < delta):
            rampslistB = RampsList()
        else:
            a0new = (vp - v0)/t0new
            if (abs(a0new) > am + EPSILON):
                rampslistB = RampsList()
            else:
                ramp1B = Ramp(v0, a0new, t0new, x0 = x0)
                rampslistB = RampsList([ramp1B, lastramp])
    
    """
    C
    - fix according to PP1
    """
    rampslistC =  PP_1(rampslist, vm, am, delta)
    if (abs(rampslistC[0].finalv) > vm):
        rampslistC = RampsList()
                
    if (rampslistA.isvoid):
        if (rampslistB.isvoid):
            if (rampslistC.isvoid):
                ## this cannot happen
                print "[PLP4::warning] no solution"
                raw_input()
                return RampsList()
            else:
                if PRINTCASE:
                    print CASEC
                return rampslistC
        else:
            if (rampslistC.isvoid):
                if PRINTCASE:
                    print CASEB
                return rampslistB
            else:
                if (rampslistB.duration <= rampslistC.duration):
                    if PRINTCASE:
                        print CASEB
                    return rampslistB
                else:
                    if PRINTCASE:
                        print CASEC
                    return rampslistC
    else:
        if (rampslistB.isvoid):
            if (rampslistC.isvoid):
                if PRINTCASE:
                    print CASEA
                return rampslistA
            else:
                if (rampslistA.duration <= rampslistC.duration):
                    if PRINTCASE:
                        print CASEA
                    return rampslistA
                else:
                    if PRINTCASE:
                        print CASEC
                    return rampslistC
        else:
            if (rampslistC.isvoid):
                if (rampslistA.duration <= rampslistB.duration):
                    if PRINTCASE:
                        print CASEA
                    return rampslistA
                else:
                    if PRINTCASE:
                        print CASEB
                    return rampslistB
            else:
                if ((rampslistA.duration <= rampslistB.duration) and 
                    (rampslistA.duration <= rampslistC.duration)):
                    if PRINTCASE:
                        print CASEA
                    return rampslistA
                elif ((rampslistB.duration <= rampslistA.duration) and 
                      (rampslistB.duration <= rampslistC.duration)):
                    if PRINTCASE:
                        print CASEB
                    return rampslistB
                else:
                    if PRINTCASE:
                        print CASEC
                    return rampslistC


def PLP_5(rampslist, vm, am, delta):
    firstramp = copy.deepcopy(rampslist[0])
    middleramp = copy.deepcopy(rampslist[1])
    lastramp = copy.deepcopy(rampslist[2])

    t0 = firstramp.T
    t1 = middleramp.T
    t2 = lastramp.T
    
    v0 = firstramp.v
    v1 = lastramp.Evald(t2)
    
    CASEA = "PLP5A"
    CASEB = "PLP5B"
    CASEC = "PLP5C"
    
    x0 = firstramp.x0
    d = rampslist.distance
    vp = middleramp.v
    
    """
    A
    - stretch the last ramp
    - fix the remaining distance with InterpolateArbitraryVel1DWithDelta
    """
    a2new = (v1 - vp)/delta
    ramp3A = Ramp(vp, a2new, delta)
    d_rem = d - ramp3A.distance
    res = InterpolateArbitraryVel1DWithDelta(0, d_rem, v0, vp, vm, am, delta)
    if (res.isvoid):
        rampslistA = RampsList()
    elif (len(res) == 1):
        res[0].x0 = x0
        rampslistA = RampsList([res[0], ramp3A])
    elif (len(res) == 2):
        res[0].x0 = x0
        rampslistA = RampsList([res[0], res[1], ramp3A])

    """
    B
    - merge the last two ramps together
    - the original first ramp remains the same 
    """
    d_rem = d - firstramp.distance
    if (abs(vp + v1) < EPSILON):
        rampslistB = RampsList()
    else:
        t1new = (2*d_rem)/(vp + v1)
        if (t1new < delta):
            rampslistB = RampsList()
        else:
            a1new = (v1 - vp)/t1new
            if (abs(a1new) > am + EPSILON):
                rampslistB = RampsList()
            else:
                ramp2B = Ramp(vp, a1new, t1new)
                rampslistB = RampsList([firstramp, ramp2B])
        
    """
    C
    - fix according to PP2
    """
    rampslistC =  PP_2(rampslist, vm, am, delta)
    if (abs(rampslistC[0].finalv) > vm):
        rampslistC = RampsList()
                
    if (rampslistA.isvoid):
        if (rampslistB.isvoid):
            if (rampslistC.isvoid):
                ## this cannot happen
                print "[PLP5::warning] no solution"
                raw_input()
                return RampsList()
            else:
                if PRINTCASE:
                    print CASEC
                return rampslistC
        else:
            if (rampslistC.isvoid):
                if PRINTCASE:
                    print CASEB
                return rampslistB
            else:
                if (rampslistB.duration <= rampslistC.duration):
                    if PRINTCASE:
                        print CASEB
                    return rampslistB
                else:
                    if PRINTCASE:
                        print CASEC
                    return rampslistC
    else:
        if (rampslistB.isvoid):
            if (rampslistC.isvoid):
                if PRINTCASE:
                    print CASEA
                return rampslistA
            else:
                if (rampslistA.duration <= rampslistC.duration):
                    if PRINTCASE:
                        print CASEA
                    return rampslistA
                else:
                    if PRINTCASE:
                        print CASEC
                    return rampslistC
        else:
            if (rampslistC.isvoid):
                if (rampslistA.duration <= rampslistB.duration):
                    if PRINTCASE:
                        print CASEA
                    return rampslistA
                else:
                    if PRINTCASE:
                        print CASEB
                    return rampslistB
            else:
                if ((rampslistA.duration <= rampslistB.duration) and 
                    (rampslistA.duration <= rampslistC.duration)):
                    if PRINTCASE:
                        print CASEA
                    return rampslistA
                elif ((rampslistB.duration <= rampslistA.duration) and 
                      (rampslistB.duration <= rampslistC.duration)):
                    if PRINTCASE:
                        print CASEB
                    return rampslistB
                else:
                    if PRINTCASE:
                        print CASEC
                    return rampslistC


def PLP_6(rampslist, vm, am, delta):
    firstramp = copy.deepcopy(rampslist[0])
    middleramp = copy.deepcopy(rampslist[1])
    lastramp = copy.deepcopy(rampslist[2])

    t0 = firstramp.T
    t1 = middleramp.T
    t2 = lastramp.T
    
    v0 = firstramp.v
    v1 = lastramp.finalv
    
    CASEA = "PLP6A"
    CASEB = "PLP6B"
    CASEC = "PLP6C"
    CASED = "PLP6D"
    
    x0 = firstramp.x0
    d = rampslist.distance
    vp = middleramp.v
    
    newfirstramp = Ramp(v0, (vp - v0)/delta, delta, x0 = x0)
    newlastramp = Ramp(vp, (v1 - vp)/delta, delta)

    """
    A
    - stretch both the first and the last ramps
    - the middle ramp is shortened
    - if the middle ramp is longer than delta, then return this case
    """
    d_rem = d - (newfirstramp.distance + newlastramp.distance)
    t1 = d_rem/vp
    if (t1 < 0):
        rampslistA = RampsList()
    else:
        newmiddleramp = Ramp(vp, 0.0, t1)
        temp = RampsList([newfirstramp, newmiddleramp, newlastramp])
        if (t1 < delta):
            rampslistA = FixSwitchTimeArbitraryVel_PLP(temp, vm, am, delta)
        else:
            rampslistA = temp

    """
    B
    - stretch only the first ramp
    """
    d_rem = d - newfirstramp.distance
    newlastrampB = InterpolateArbitraryVel1DWithDelta(0, d_rem, vp, v1, vm, am, delta)
    if (len(newlastrampB) == 2):
        rampslistB1 = RampsList([newfirstramp, newlastrampB[0], newlastrampB[1]])
    else:
        rampslistB1 = RampsList([newfirstramp, newlastrampB[0]])

    rampslistB2 = PP_1(rampslist, vm, am, delta)
    if (not CheckRampsList(rampslistB2, vm, am, delta)):
        rampslistB = rampslistB1
    else:        
        if (rampslistB1.duration < rampslistB2.duration):
            rampslistB = rampslistB1
        else:
            rampslistB = rampslistB2
        
    """
    C
    - stretch only the last ramp
    """
    d_rem = d - newlastramp.distance
    newfirstrampC = InterpolateArbitraryVel1DWithDelta(0, d_rem, v0, vp, vm, am, delta)
    newfirstrampC[0].x0 = x0
    if (len(newfirstrampC) == 2):
        rampslistC1 = RampsList([newfirstrampC[0], newfirstrampC[1], newlastramp])
    else:
        rampslistC1 = RampsList([newfirstrampC[0], newlastramp])
        
    rampslistC2 = PP_2(rampslist, vm, am, delta)
    if (not CheckRampsList(rampslistC2, vm, am, delta)):
        rampslistC = rampslistC1
    else:
        if (rampslistC1.duration < rampslistC2.duration):
            rampslistC = rampslistC1
        else:
            rampslistC = rampslistC2
        
    """
    D
    - sometimes the first and the last ramps are too short
    - merge everything into one ramp
    """
    if (abs(v0 + v1) < EPSILON):
        rampslistD = RampsList()
    tnew = (2*d)/(v0 + v1)
    if (tnew < 0):
        rampslistD = RampsList()
    else:
        anew = (v1 - v0)/tnew
        ramp1D = Ramp(v0, anew, tnew, x0 = x0)
        rampslistD = RampsList([ramp1D])       
        
    if not (rampslistA.isvoid):
        if (rampslistA.duration <= rampslistB.duration) \
                and (rampslistA.duration <= rampslistC.duration) \
                and (rampslistA[1].T >= delta):
            if not (rampslistD.isvoid):
                if (rampslistA.duration <= rampslistD.duration):
                    if PRINTCASE:
                        print CASEA
                    return rampslistA
            else:
                if PRINTCASE:
                    print CASEA
                return rampslistA
        
    if (rampslistB.duration <= rampslistC.duration):
        if not (rampslistD.isvoid):
            if (rampslistB.duration <= rampslistD.duration):
                if PRINTCASE:
                    print CASEB
                return rampslistB
        else:
            if PRINTCASE:
                print CASEB
            return rampslistB

    if (rampslistC.duration <= rampslistB.duration):
        if not (rampslistD.isvoid):
            if (rampslistC.duration <= rampslistD.duration):
                if PRINTCASE:
                    print CASEC
                return rampslistC
        else:
            if PRINTCASE:
                print CASEC
            return rampslistC
                    
    if PRINTCASE:
        print CASED
    return rampslistD
    

def PLP_7(rampslist, vm, am, delta):
    firstramp = copy.deepcopy(rampslist[0])
    middleramp = copy.deepcopy(rampslist[1])
    lastramp = copy.deepcopy(rampslist[2])

    t0 = firstramp.T
    t1 = middleramp.T
    t2 = lastramp.T
    
    v0 = firstramp.v
    v1 = lastramp.finalv
    
    CASEA = "PLP7A"
    CASEB = "PLP7B"
    CASEC = "PLP7C"
    
    x0 = firstramp.x0
    d = rampslist.distance
    vp = middleramp.v
    
    """
    A
    - modify to (delta, delta, delta)
    """
    rampslistA = Reinterpolate1D_ThreeRamps(rampslist, delta, delta, delta, vm, am)
    
    
    """
    B
    - fix using PP3
    """
    rampslistB =  PP_3(rampslist, vm, am, delta)
        
    
    """
    C
    - exceptional case where (neccessary condition)
        -- one of terminal velocities is zero
        (-- the distance is of the opposite sign to the non-zero velocity)
    - stretch the longest ramp to delta
    - the remaining part is merged into one ramp
    """
    if (abs(v0) < EPSILON) or (abs(v1) < EPSILON):
        if (abs(vp - v0) > abs(vp - v1)):
            dnew = 0.5*(vp + v0)*delta
            t1new = (2*d - 2*dnew)/(vp + v1)
            if (t1new > 0):
                a0new = (vp - v0)/delta
                ramp1C = Ramp(v0, a0new, delta, x0 = x0)
                a1new = (v1 - vp)/t1new
                ramp2C = Ramp(vp, a1new, t1new)
                rampslistC = RampsList([ramp1C, ramp2C])
            else:
                rampslistC = RampsList()
        else:
            dnew = 0.5*(vp + v1)*delta
            t0new = (2*d - 2*dnew)/(vp + v0)
            if (t0new > 0):
                a0new = (vp - v0)/t0new
                ramp1C = Ramp(v0, a0new, t0new, x0 = x0)
                a1new = (v1 - vp)/delta
                ramp2C = Ramp(vp, a1new, delta)
                rampslistC = RampsList([ramp1C, ramp2C])
            else:
                rampslistC = RampsList()
                
        if (not CheckRampsList(rampslistC, vm, am, delta)):
            rampslistC = RampsList()
    else:
        rampslistC = RampsList()

    if (rampslistA.isvoid):
        if (rampslistB.isvoid):
            if (rampslistC.isvoid):
                ## this cannot happen
                print "[PLP7::warning] no solution"
                raw_input()
                return RampsList()
            else:
                if PRINTCASE:
                    print CASEC
                return rampslistC
        else:
            if (rampslistC.isvoid):
                if PRINTCASE:
                    print CASEB
                return rampslistB
            else:
                if (rampslistB.duration <= rampslistC.duration):
                    if PRINTCASE:
                        print CASEB
                    return rampslistB
                else:
                    if PRINTCASE:
                        print CASEC
                    return rampslistC
    else:
        if (rampslistB.isvoid):
            if (rampslistC.isvoid):
                if PRINTCASE:
                    print CASEA
                return rampslistA
            else:
                if (rampslistA.duration <= rampslistC.duration):
                    if PRINTCASE:
                        print CASEA
                    return rampslistA
                else:
                    if PRINTCASE:
                        print CASEC
                    return rampslistC
        else:
            if (rampslistC.isvoid):
                if (rampslistA.duration <= rampslistB.duration):
                    if PRINTCASE:
                        print CASEA
                    return rampslistA
                else:
                    if PRINTCASE:
                        print CASEB
                    return rampslistB
            else:
                if ((rampslistA.duration <= rampslistB.duration) and 
                    (rampslistA.duration <= rampslistC.duration)):
                    if PRINTCASE:
                        print CASEA
                    return rampslistA
                elif ((rampslistB.duration <= rampslistA.duration) and 
                      (rampslistB.duration <= rampslistC.duration)):
                    if PRINTCASE:
                        print CASEB
                    return rampslistB
                else:
                    if PRINTCASE:
                        print CASEC
                    return rampslistC


## Re-interpolating trajectories with a fixed time Tmax
def ReinterpolateND_FixedDuration(rampslistnd0, vmvect, amvect, delta, maxindex = -1):
    ndof = len(rampslistnd0)
    if (maxindex < 0):
        Tmax = 0
        maxindex = 0
        for j in range(ndof):
            if (rampslistnd0[j].duration > Tmax):
                Tmax = rampslistnd0[j].duration
                maxindex = j
    else:
        Tmax = rampslistnd0[maxindex].duration
    
    rampslistnd1 = []
    
    CASE = "ReinterpolateND_FixedDuration"
        
    ## compute grids on the slowest trajectory
    grid = ComputeGrids(rampslistnd0[maxindex], delta)
    
    for j in range(ndof):
        if (j == maxindex):
            rampslistnd1.append(rampslistnd0[j])
            continue        
        if (len(grid) > GRID_THRESHOLD):
            rampslist = SnapToGrids_TwoRamps(rampslistnd0[j], vmvect[j], amvect[j], 
                                             delta, Tmax, grid)
            if (rampslist.isvoid):
                rampslist = SnapToGrids_ThreeRamps(rampslistnd0[j], vmvect[j], amvect[j],
                                                   delta, Tmax, grid)
                if (rampslist.isvoid):                
                    if PRINTCASE:
                        print CASE + ": DOF {0} is not feasible".format(j)
                    return RampsListND()
                else:
                    rampslistnd1.append(rampslist)
            else:
                rampslistnd1.append(rampslist)
        else:
            rampslist = SnapToGrids_ThreeRamps(rampslistnd0[j], vmvect[j], amvect[j], 
                                               delta, Tmax, grid)
            if (rampslist.isvoid):
                rampslist = SnapToGrids_TwoRamps(rampslistnd0[j], vmvect[j], amvect[j],
                                                 delta, Tmax, grid)
                if (rampslist.isvoid):                
                    if PRINTCASE:
                        print CASE + ": DOF {0} is not feasible".format(j)
                    return RampsListND()
                else:
                    rampslistnd1.append(rampslist)
            else:
                rampslistnd1.append(rampslist)
            
    return RampsListND(rampslistnd1)


## ComputeGrids returns grids that divide each ramp of the trajectory
## into equally long segment which is the smallest but still longer
## than \delta
def ComputeGrids(rampslist, delta):
    T = rampslist.duration
    
    if (delta < EPSILON):
        ## no minimum-switch-time constraint
        grid = np.linspace(0, T, DEFAULT_NUM_GRID + 2)[1:(DEFAULT_NUM_GRID + 1)]
        return grid.tolist()

    if (len(rampslist) == 1):
        n0 = min(np.floor(T/delta), DEFAULT_NUM_GRID)
        grid = np.linspace(0, T, n0 + 2)[1:(n0 + 1)]
        return grid.tolist()
    elif (len(rampslist) == 2):
        t0 = rampslist[0].T
        t1 = rampslist[1].T
        n0 = np.floor(t0/delta)
        n1 = np.floor(t1/delta)
        if ((n0 + n1 + 1) > DEFAULT_NUM_GRID):
            ## the number of grids exceeds DEFAULT_NUM_GRID, reduce the
            ## number of grids to DEFAULT_NUM_GRID distribute grids
            ## according to the duration of each ramp
            m0 = n0 # temporary var
            m1 = n1 # temporary var
            n0 = np.floor((m0/(m0 + m1))*(DEFAULT_NUM_GRID + 1))
            n1 = np.floor((m1/(m0 + m1))*(DEFAULT_NUM_GRID + 1))

        grid0 = np.linspace(0, t0, n0 + 1)[1:n0]
        grid1 = np.linspace(t0, t0 + t1, n1 + 1)[0:n1]
        grid = np.concatenate((grid0, grid1))
        return grid.tolist()
            
    else:
        t0 = rampslist[0].T
        t1 = rampslist[1].T
        t2 = rampslist[2].T
        n0 = np.floor(t0/delta)
        n1 = np.floor(t1/delta)
        n2 = np.floor(t2/delta)
        if ((n0 + n1 + n2 + 2) > DEFAULT_NUM_GRID):
            ## the number of grids exceeds DEFAULT_NUM_GRID. reduce the
            ## number of grids to DEFAULT_NUM_GRID distribute grids
            ## according to the duration of each ramp
            m0 = n0 # temporary var
            m1 = n1 # temporary var
            m2 = n2 # temporary var
            n0 = np.floor((m0/(m0 + m1 + m2))*(DEFAULT_NUM_GRID + 1))
            n1 = np.floor((m1/(m0 + m1 + m2))*(DEFAULT_NUM_GRID + 1))
            n2 = np.floor((m2/(m0 + m1 + m2))*(DEFAULT_NUM_GRID + 1))
            ## remove one grid from the shortest ramp
            if (t0 < t1) and (t0 < t2):
                n0 -= 1
            elif (t1 < t0) and (t1 < t2):
                n1 -= 1
            else:
                n2 -= 1
        grid0 = np.linspace(0, t0, n0 + 1)[1:n0]
        grid1 = np.linspace(t0, t0 + t1, n1 + 1)[0:n1]
        grid2 = np.linspace(t0 + t1, t0 + t1 + t2, n2 + 1)[0:n2]
        grid = np.concatenate((grid0, grid1, grid2))
        return grid.tolist()
    

## SnapToGrids_TwoRamps re-interpolates a single DOF trajectory to be a
## PP-trajectory with the switch point lying on one of grids.
def SnapToGrids_TwoRamps(rampslist0, vm, am, delta, Tmax, grid):
    for g in grid:
        assert(g > 0)
        rampslist = Reinterpolate1D_TwoRamps(rampslist0, g, Tmax - g)
        if CheckRampsList(rampslist, vm, am, delta):
            return rampslist
    return RampsList()


## SnapToGrids_ThreeRamps re-interpolates a single DOF trajectory to be a
## PLP-trajectory.
def SnapToGrids_ThreeRamps(rampslist0, vm, am, delta, Tmax, grid):
    ## make an initial guess for t0, t1, and t2
    t0 = Tmax*0.25
    t1 = Tmax*0.50
    
    index1 = bisect.bisect_left(grid, t0)
    # i1 contains grid(s) for the first switch point to snap
    if index1 == 0:
        i1 = [index1]
    else:
        i1 = [index1 - 1, index1]

    index2 = bisect.bisect_left(grid, t0 + t1)
    # i2 contains grid(s) for the second switch point to snap
    if index2 == len(grid):
        i2 = [index2 - 1]
    else:
        i2 = [index2 - 1, index2]
        
    for index1 in i1:
        for index2 in i2:
            if (index2 <= index1):
                continue
            
            t0new = grid[index1]
            t1new = grid[index2] - grid[index1]
            t2new = Tmax - grid[index2]
            rampslist = Reinterpolate1D_ThreeRamps(rampslist0, t0new, t1new, t2new,
                                                   vm, am)
            if (not (rampslist.isvoid)):
                return rampslist
    return RampsList()


## Reinterpolate1D_TwoRamps re-interpolates a trajectory to be three
## ramps according to given ramp durations t0 and t1.
def Reinterpolate1D_TwoRamps(rampslist0, t0, t1):
    assert(t0 > 0)
    assert(t1 > 0)
    
    v0 = rampslist0[0].v
    v1 = rampslist0[-1].finalv
    
    alpha = t0*(0.5*t0 + t1)
    beta = 0.5*t1*t1
    gamma = rampslist0.distance - v0*(t0 + t1)
    det = alpha*t1 - beta*t0
    
    a0 = (gamma*t1 - beta*(v1 - v0))/det
    a1 = (-gamma*t0 + alpha*(v1 - v0))/det
    
    ramp1 = Ramp(v0, a0, t0, x0 = rampslist0.x0)
    ramp2 = Ramp(v0 + a0*t0, a1, t1)
    
    return RampsList([ramp1, ramp2])


## Reinterpolate1D_ThreeRamps re-interpolates a trajectory to be three
## ramps according to given ramp durations t0, t1, and t2.
def Reinterpolate1D_ThreeRamps(rampslist0, t0, t1, t2, vm, am):
    assert(t0 > 0)
    assert(t1 > 0)
    assert(t2 > 0)

    v0 = rampslist0[0].v
    v1 = rampslist0[-1].finalv
    alpha = t0*(0.5*t0 + t1 + t2)
    beta = t1*(0.5*t1 + t2)
    sigma = 0.5*t2*t2
    gamma = rampslist0.distance - v0*(t0 + t1 + t2)
    kappa = v1 - v0
    
    A = np.array([[alpha, beta, sigma], [t0, t1, t2]])
    b = np.array([[gamma], [kappa]])
    AAT = np.dot(A, A.T)
    pseudoinvA = np.dot(A.T, np.linalg.inv(AAT))
    xp = np.dot(pseudoinvA, b) # particular solution
    xh = np.array([[(beta*t2 - sigma*t1)/(alpha*t1 - beta*t0)],
                   [-(alpha*t2 - sigma*t0)/(alpha*t1 - beta*t0)],
                   [1.0]]) # homogenous solution
    
    l = np.array([[(max((-vm - v0)/t0, -am))], [-am], [max((-vm + v1)/t2, -am)]])
    u = np.array([[min((vm - v0)/t0, am)], [am], [min((vm + v1)/t2, am)]])
    
    interval0 = SolveIneq(xh[0], xp[0], l = l[0], u = u[0])
    interval1 = SolveIneq(xh[1], xp[1], l = l[1], u = u[1])
    interval2 = SolveIneq(xh[2], xp[2], l = l[2], u = u[2])
    
    result = Intersect(interval0, interval1)
    if result.isvoid:
        return RampsList()
    result = Intersect(result, interval2)
    if result.isvoid:
        return RampsList()
    ## choose one value of k to construct a solution
    k = RNG.uniform(result.l, result.u)

    x = xp + k*xh
    a0 = x[0]
    a1 = x[1]
    a2 = x[2]
    ramp1 = Ramp(v0, a0, t0, rampslist0.x0)
    ramp2 = Ramp(ramp1.Evald(t0), a1, t1)
    ramp3 = Ramp(ramp2.Evald(t1), a2, t2)
    
    return RampsList([ramp1, ramp2, ramp3])


## CheckRampsList checks whether the input rampslist violates any constraint
def CheckRampsList(rampslist, vm, am, delta, reframpslist = RampsList(), PRINT = False):
    if (rampslist.isvoid):
        return False
    
    m = len(rampslist)
    if PRINT:
        print "\n******************************"
        print "Rampslist has {0} ramps".format(m)
        
    isfeasible = True
        
    for i in range(m):
        if (rampslist[i].T < 0) or (rampslist[i].T > INF):
            print "ramp {0} has an invalid duration".format(i)
            return False
        
        cond1 = abs(rampslist[i].v) <= vm + EPSILON
        cond2 = abs(rampslist[i].a) <= am + EPSILON
        cond3 = abs(rampslist[i].T) >= delta - EPSILON
        isfeasible = isfeasible and cond1 and cond2 and cond3
        
        if PRINT:
            print "  checking ramp {0}".format(i)
            print "    velocity constraint           : {0}".format(Colorize(cond1))
            print "    acceleration constraint       : {0}".format(Colorize(cond2))
            print "    minimum-switch-time constraint: {0}".format(Colorize(cond3))
            
        if not isfeasible:
            return False
        
    if not (reframpslist.isvoid):
        cond4 = abs(rampslist.distance - reframpslist.distance) < EPSILON
        isfeasible = isfeasible and cond4
        if PRINT:
            print "  checking distance with ref. rampslist:: {0}".format(Colorize(cond4))
            
    return isfeasible


##
def Colorize(status):
    if status:
        return "\033[1;32mpassed\033[0m"
    else:
        return "\033[1;31mfailed\033[0m"
