import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import bisect
from pylab import ion
ion()

from utilities import epsilon, inf, FuzzyEquals, FuzzyZero


class Ramp(object):
    """A Ramp is a constant-acceleration one-dimensional trajectory. When plotting its velocity
    evolution over time, the graph is a `ramp` on the velocity-time plane, hence the name.

    Parameters (input)
    ------------------
    v0 : float
        Initial velocity of the trajectory.
    a : float
        Acceleration of the trajectory.
    duration : float
        Duration of the trajectory. It must be non-negative.
    x0 : float, optional
        Initial displacement of the trajectory. If not given, x0 will be set to zero.

    Parameters (calculated from inputs)
    -----------------------------------
    v1 : float
        Final velocity of the trajectory.
    d : float
        total displacement made by this Ramp (i.e., independent of x0).
    x1 : float
        Final displacement at the end of the trajectory: x1 = x0 + d.

    """
    def __init__(self, v0, a, t, x0=0):
        assert(t >= -epsilon)

        self.v0 = v0
        self.a = a
        self.duration = t
        self.x0 = x0

        self.v1 = v0 + self.a*self.duration
        self.d = self.duration*(self.v0 + 0.5*self.a*self.duration)
        self.x1 = self.x0 + self.d


    def Initialize(self, v0, a, t, x0=0):
        """Initialize (or reinitialize) the Ramp with the given parameters and calculated other parameters
        accordingly.

        Parameters (input)
        ------------------
        v0 : float
            Initial velocity of the trajectory.
        a : float
            Acceleration of the trajectory.
        duration : float
            Duration of the trajectory. It must be non-negative.
        x0 : float, optional
            Initial displacement of the trajectory. If not given, x0 will be set to zero.

        """
        assert(t >= -epsilon)

        self.v0 = v0
        self.a = a
        self.duration = t
        self.x0 = x0

        self.v1 = v0 + self.a*self.duration
        self.d = self.duration*(self.v0 + 0.5*self.a*self.duration)
        self.x1 = self.x0 + self.d


    def EvalPos(self, t):
        """Evalutaion the position at the given time instant.

        Parameters
        ----------
        t : float
            Time instant at which to evaluate the position.

        Returns
        -------
        x : float
            Position at time t.

        """
        if (t <= 0):
            return self.x0
        elif (t >= self.duration):
            return self.x1
        else:
            return self.x0 + t*(self.v0 + 0.5*self.a*t)


    def EvalVel(self, t):
        """Evalutaion the velocity at the given time instant.

        Parameters
        ----------
        t : float
            Time instant at which to evaluate the velocity.

        Returns
        -------
        v : float
            Velocity at time t.

        """
        if (t <= 0):
            return self.v0
        elif (t >= self.duration):
            return self.v1
        else:
            return self.v0 + self.a*t

        
    def EvalAcc(self):
        """Return the acceleration.
        
        Returns
        -------
        a : float
            Acceleration.

        """
        return self.a


    def GetPeaks(self):
        """Calculate the peaks of positions along the trajectory.

        Returns
        -------
        xmin : float
            Minimum position along the trajectory.
        xmax : float
            Maximum position along the trajectory.

        """
        return self._GetPeaks(0, self.duration)


    def _GetPeaks(self, ta, tb):
        if (ta > tb):
            return self._GetPeaks(tb, ta)

        if (ta < 0):
            ta = 0
        if (tb <= 0):
            return [self.x0, self.x0]

        if (tb > self.duration):
            tb = self.duration
        if (ta >= self.duration):
            return [self.x1, self.x1]

        curMin = self.EvalPos(ta)
        curMax = self.EvalPos(tb)
        if curMin > curMax:
            [curMin, curMax] = Swap(curMax, curMin)
            
        if FuzzyZero(self.a, epsilon):
            return [curMin, curMax]

        tDeflection = -self.v0/self.a
        if (tDeflection <= ta) or (tDeflection >= tb):
            return [curMin, curMax]

        xDeflection = self.x0 + 0.5*self.v0*tDeflection
        curMin = min(curMin, xDeflection)
        curMax = max(curMax, xDeflection)
        return [curMin, curMax]


    def SetInitialValue(self, newX0):
        """Set the initial displacement (position) of the trajectory to the given value and recalculate the
        related parameters accordingly.

        Parameters
        ----------
        newX0 : float
            The new initial displacement.

        """
        self.x0 = newX0
        self.x1 = self.x0 + self.d


    def UpdateDuration(self, newDuration):
        """Set the trajectory duration to the given value and recalculate the related parameters accordingly.

        Parameters
        ----------
        newDuration : float
            The new duration.
        
        """
        assert(newDuration >= -epsilon)
        if (newDuration <= 0):
            self.duration = 0
            self.x1 = self.x0
            self.v1 = self.v0
            self.d = 0
            return

        self.duration = newDuration
        self.v1 = self.v0 + self.a*self.duration
        self.d = self.duration*(self.v0 + 0.5*self.a*self.duration)
        self.x1 = self.x0 + self.d


    def PlotPos(self, t0=0, fignum=None, **kwargs):
        """Plot trajectory position vs time.

        Parameters
        ----------
        t0 : float
            Position on the horizontal (time) axis to start plotting.
        fignum : int, float, string
            Figure's number/title
        
        """
        raise NotImplementedError


    def PlotVel(self, t0=0, fignum=None, **kwargs):
        """Plot trajectory velocity vs time.

        Parameters
        ----------
        t0 : float
            Position on the horizontal (time) axis to start plotting.
        fignum : int, float, string
            Figure's number/title
        
        """
        if fignum is not None:
            plt.figure(fignum)

        line = plt.plot([t0, t0 + self.duration], [self.v0, self.v1], **kwargs)[0]
        plt.show(False)
        return line


    def PlotAcc(self, t0=0, fignum=None, **kwargs):
        """Plot trajectory acceleration vs time.

        Parameters
        ----------
        t0 : float
            Position on the horizontal (time) axis to start plotting.
        fignum : int, float, string
            Figure's number/title
        
        """
        if fignum is not None:
            plt.figure(fignum)

        line = plt.plot([t0, t0 + self.duration], [self.a, self.a], **kwargs)[0]
        plt.show(False)
        return line


    def Cut(self, t):
        """Cut the trajectory into two halves. self will be the left half which is the segment from time 0
        to t. The other half (segment from time t to duration) is returned.

        Parameters
        ----------
        t : float
            Time instant at which to cut the trajectory

        Returns
        -------
        remRamp : Ramp
            Segment from time t to duration of the original trajectory.

        """
        if (t <= 0):
            remRamp = Ramp(self.v0, self.a, self.duration, self.x0)
            self.Initialize(self.v0, 0, 0, self.x0)
            return remRamp
        elif (t >= self.duration):
            renRamp = Ramp(self.v1, 0, 0, self.x1)
            return remRamp

        remRampDuration = self.duration - t
        self.UpdateDuration(t)
        remRamp = Ramp(self.v1, self.a, remRampDuration, self.x1)
        return remRamp


    def TrimFront(self, t):
        """Trim out the trajectory segment from time 0 to t.

        Parameters
        ----------
        t : float
            Time instant at which to trim the trajectory
        
        """
        if (t <= 0):
            return
        elif (t >= self.duration):
            self.Initialize(self.v1, 0, 0, self.x1)
            return

        remDuration = self.duration - t
        newX0 = self.EvalPos(t)
        newV0 = self.EvalVel(t)
        self.Initialize(newV0, self.a, remDuration, newX0)
        return


    def TrimBack(self, t):
        """Trim out the trajectory segment from time t to duration.

        Parameters
        ----------
        t : float
            Time instant at which to trim the trajectory
        
        """
        if (t <= 0):
            self.Initialize(self.v0, 0, 0, self.x0)
            return
        elif (t >= self.duration):
            return

        self.UpdateDuration(t)
        return
        

class ParabolicCurve(object):
    """A ParabolicCurve is a piecewise-constant-acceleration one-dimensional trajectory. It is a
    concatenation of Ramps.

    Parameters (input)
    ------------------
    ramps : list of Ramps, optional
        The list of Ramps to construct the ParabolicCurve with.

    Parameters (generated from inputs)
    ----------------------------------
    x0 : float
        Initial displacement of the trajectory.
    x1 : float
        Final displacement of the trajectory.
    v0 : float
        Initial velocity of the trajectory.
    v1 : float
        Final velocity of the trajectory.
    switchpointsList : list of float
        List of switch points, time instants at which the acceleration changes.
    duration : float
        Duration of the trajectory.
    d : float
        Total displacement made by this trajectory.

    """
    def __init__(self, ramps=[]):
        self.switchpointsList = []
        dur = 0.0
        d = 0.0
        self.ramps = []
        
        if len(ramps) == 0:
            self.isEmpty = True
            self.x0 = 0.0
            self.x1 = 0.0
            self.v0 = 0.0
            self.v1 = 0.0
            self.switchpointsList = []
            self.duration = dur
            self.d = d
        else:
            self.ramps = deepcopy(ramps)
            self.isEmpty = False
            self.v0 = self.ramps[0].v0
            self.v1 = self.ramps[-1].v1
            
            self.switchpointsList.append(dur)
            for ramp in self.ramps:
                dur += ramp.duration
                self.switchpointsList.append(dur)
                d += ramp.d
            self.duration = dur
            self.d = d

            self.SetInitialValue(self.ramps[0].x0)


    def __getitem__(self, index):
        return self.ramps[index]


    def __len__(self):
        return len(self.ramps)


    def Initialize(self):
        self.switchpointsList = []
        dur = 0.0
        d = 0.0
        self.ramps = []
        
        if len(ramps) == 0:
            self.x0 = 0.0
            self.x1 = 0.0
            self.v0 = 0.0
            self.v1 = 0.0
            self.switchpointsList = []
            self.duration = dur
            self.d = d
        else:
            self.ramps = deepcopy(ramps)
            self.v0 = self.ramps[0].v0
            self.v1 = self.ramps[-1].v1
            
            self.switchpointsList.append(dur)
            for ramp in self.ramps:
                dur += ramp.duration
                self.switchpointsList.append(dur)
                d += ramp.d
            self.duration = dur
            self.d = d

            self.SetInitialValue(self.ramps[0].x0)


    def IsEmpty(self):
        return len(self) == 0


    def Append(self, curve):
        """Append a ParabolicCurve to this one.

        Parameters
        ----------
        curve : ParabolicCurve
            ParabolicCurve to be appended.
        
        """
        if len(self) == 0:
            if len(curve) > 0:
                self.ramps = deepcopy(curve.ramps)
                self.x0 = curve.x0
                self.x1 = curve.x1
                self.v0 = curve.v0
                self.v1 = curve.v1
                self.duration = curve.duration
                self.d = curve.d
                self.switchpointsList = deepcopy(curve.switchpointsList)
            else:
                pass
            return
        else:
            dur = self.duration
            d = self.d
            for ramp in curve.ramps:
                self.ramps.append(deepcopy(ramp))
                # Update displacement for the newly appended ramp
                self.ramps[-1].SetInitialValue(self.ramps[-2].x1)
                d += ramp.d
                dur += ramp.duration
                self.switchpointsList.append(dur)

            self.v1 = self.ramps[-1].v1
            self.duration = dur
            self.d = d
            self.x1 = self.x0 + self.d


    def FindRampIndex(self, t):
        """Find the index of the ramp in which the given time instant lies.

        Parameters
        ----------
        t : float
            Time instant.
        
        Returns
        -------
        i : int
            Ramp index.
        remainder : float
            Time interval between the beginning of the ramp to t.
        
        """
        if (t <= epsilon):
            i = 0
            remainder = 0.0
        else:
            i = bisect.bisect_left(self.switchpointsList, t) - 1
            remainder = t - self.switchpointsList[i]
        return [i, remainder]


    def EvalPos(self, t):
        """Evalutaion the position at the given time instant.

        Parameters
        ----------
        t : float
            Time instant at which to evaluate the position.

        Returns
        -------
        x : float
            Position at time t.

        """
        assert(t >= -epsilon)
        assert(t <= self.duration + epsilon)

        i, remainder = self.FindRampIndex(t)
        return self.ramps[i].EvalPos(remainder)


    def EvalVel(self, t):
        """Evalutaion the velocity at the given time instant.

        Parameters
        ----------
        t : float
            Time instant at which to evaluate the velocity.

        Returns
        -------
        v : float
            Velocity at time t.

        """
        assert(t >= -epsilon)
        assert(t <= self.duration + epsilon)

        i, remainder = self.FindRampIndex(t)
        return self.ramps[i].EvalVel(remainder)


    def EvalAcc(self, t):
        """Evalutaion the acceleration at the given time instant.

        Parameters
        ----------
        t : float
            Time instant at which to evaluate the acceleration.

        Returns
        -------
        a : float
            Acceleration at time t.

        """
        assert(t >= -epsilon)
        assert(t <= self.duration + epsilon)

        i, remainder = self.FindRampIndex(t)
        return self.ramps[i].EvalAcc(remainder)


    def GetPeaks(self):
        """Calculate the peaks of positions along the trajectory.

        Returns
        -------
        xmin : float
            Minimum position along the trajectory.
        xmax : float
            Maximum position along the trajectory.

        """
        return self._GetPeaks(0, self.duration)


    def _GetPeaks(self, ta, tb):
        xmin = inf
        xmax = -inf

        for ramp in self.ramps:
            [bmin, bmax] = ramp.GetPeaks()
            if bmin < xmin:
                xmin = bmin
            if bmax > xmax:
                xmax = bmax

        return [xmin, xmax]


    def SetInitialValue(self, x0):
        """Set the initial displacement (position) of the trajectory to the given value and recalculate the
        related parameters accordingly.

        Parameters
        ----------
        newX0 : float
            The new initial displacement.

        """
        self.x0 = x0
        newX0 = x0
        for ramp in self.ramps:
            ramp.SetInitialValue(newX0)
            newX0 += ramp.d
        self.x1 = self.x0 + self.d


    def SetConstant(self, x0, t):
        """Set this ParabolicCurve to be a constant trajectory (i.e. zero-velocity and zero-acceleration.

        Parameters
        ----------
        x0 : float
            New initial dispalcement of the trajectory
        t : float
            New trajectory duration
        
        """
        assert(t >= 0)
        ramp = Ramp(0, 0, t, x0)
        self.Initialize([ramp])
        return


    def SetSegment(self, x0, x1, v0, v1, t):
        assert(t >= -epsilon)
        if FuzzyZero(t, epsilon):
            a = 0
        else:
            tSqr = t*t
            a = -(v0*tSqr + t*(x0 - x1) + 2*(v0 - v1))/(t*(0.5*tSqr + 2))

        ramp = Ramp()
        ramp.x0 = x0
        ramp.x1 = x1
        ramp.v0 = v0
        ramp.v1 = v1
        ramp.duration = t
        ramp.d = x1 - x0
        ramp.a = a
        self.Initialize([ramp])
        return


    def SetZeroDuration(self, x0, v0):
        ramp = Ramp(v0, 0, 0, x0)
        self.Initialize([ramp])
        return
        
        
    def Cut(self, t):
        """Cut the trajectory into two halves. self will be the left half which contains all the ramps from
        time 0 to t. The other half of the trajectory is returned.

        Parameters
        ----------
        t : float
            Time instant at which to cut the trajectory

        Returns
        -------
        remCurve : ParabolicCurve
            ParabolicCurve containing all the ramps from time t to duration of the original trajectory.

        """
        if (t <= 0):
            remCurve = ParabolicCurve(self.ramps)
            self.SetZeroDuration(self.x0, self.v0)
            return remCurve
        elif (t >= self.duration):
            remCurve = ParabolicCurve()
            remCurve.SetZeroDuration(self.x1, self.v1)
            return

        i, remainder = self.FindRampIndex(t)
        leftHalf = self.ramps[0:i + 1]
        rightHalf = self.ramps[i:]

        leftHalf[-1].TrimBack(remainder)
        rightHalf[0].TrimFront(remainder)
        self.Initialize(leftHalf)
        remCurve = ParabolicCurve(rightHalf)
        return remCurve


    def TrimFront(self, t):
        """Trim out the trajectory segment from time 0 to t.

        Parameters
        ----------
        t : float
            Time instant at which to trim the trajectory
        
        """
        if (t <= 0):
            return
        elif (t >= self.duration):
            self.SetZeroDuration(self.x1, self.v1)
            return

        i, remainder = self.FindRampIndex(t)
        rightHalf = self.ramps[i:]

        rightHalf[0].TrimFront(remainder)
        self.Initialize(rightHalf)
        return


    def TrimBack(self, t):
        """Trim out the trajectory segment from time t to duration.

        Parameters
        ----------
        t : float
            Time instant at which to trim the trajectory
        
        """
        if (t <= 0):
            self.SetZeroDuration(self.x0, self.v0)
            return
        elif (t >= self.duration):
            return

        i, remainder = self.FindRampIndex(t)
        leftHalf = self.ramps[0:i + 1]

        leftHalf[-1].TrimBack(remainder)
        self.Initialize(leftHalf)
        return


    # Visualization
    def PlotPos(self, fignum=None, color='g', dt=0.01, lw=2, includingSW=False):
        """Plot trajectory position vs time.

        Parameters
        ----------
        fignum : int, float, string, optional
            Figure's number/title
        color : string, optional
            Matplotlib's color option.
        dt : float, optional
            Time resolution to evaluate the position.
        lw : int, optional
            Linewidth
        includingSW : bool, optional
            If True, draw vertical straight lines at switch points.
        
        """
        tVect = np.arange(0, self.duration, dt)
        if tVect[-1] < self.duration:
            tVect = np.append(tVect, self.duration)
            
        xVect = [self.EvalPos(t) for t in tVect]
        if fignum is not None:
            plt.figure(fignum)
        plt.plot(tVect, xVect, color=color, linewidth=lw)

        if includingSW:
            ax = plt.gca().axis()
            for s in self.switchpointsList:
                plt.plot([s, s], [ax[2], ax[3]], 'r', linewidth=1)
                
        plt.show(False)


    def PlotVel(self, fignum=None, color=None, lw=2, includingSW=False, **kwargs):
        """Plot trajectory velocity vs time.

        Parameters
        ----------
        fignum : int, float, string, optional
            Figure's number/title
        color : string, optional
            Matplotlib's color option.
        lw : int, optional
            Linewidth
        includingSW : bool, optional
            If True, draw vertical straight lines at switch points.
        
        """
        if fignum is not None:
            plt.figure(fignum)

        t0 = 0.0
        for ramp in self.ramps:
            if color is None:
                line = ramp.PlotVel(t0=t0, fignum=fignum, linewidth=lw, **kwargs)
                color = line.get_color()
            else:
                line = ramp.PlotVel(t0=t0, fignum=fignum, color=color, linewidth=lw, **kwargs)
            t0 += ramp.duration

        if includingSW:
            ax = plt.gca().axis()
            for s in self.switchpointsList:
                plt.plot([s, s], [ax[2], ax[3]], 'r', linewidth=1)
            
        plt.show(False)
        return line


    def PlotAcc(self, fignum=None, color=None, lw=2, **kwargs):
        """Plot trajectory velocity vs time.

        Parameters
        ----------
        fignum : int, float, string, optional
            Figure's number/title
        color : string, optional
            Matplotlib's color option.
        lw : int, optional
            Linewidth
        
        """
        if fignum is not None:
            plt.figure(fignum)
            
        t0 = 0.0
        prevAcc = self.ramps[0].a
        for ramp in self.ramps:
            if color is None:
                line = ramp.PlotAcc(t0=t0, fignum=fignum, linewidth=lw, **kwargs)
                color = line.get_color()
            else:
                line = ramp.PlotAcc(t0=t0, fignum=fignum, color=color, linewidth=lw, **kwargs)
            plt.plot([t0, t0], [prevAcc, ramp.a], color=color, linewidth=lw, **kwargs)
            t0 += ramp.duration
            prevAcc = ramp.a
        plt.show(False)
        return line


class ParabolicCurvesND(object):
    """A ParabolicCurvesND is a (parabolic) trajectory of an n-DOF system. A trajectory of each DOF is a
    ParabolicCurve.

    """
    def __init__(self, curves=[]):
        if len(curves) == 0:
            self.curves = []
            self.x0Vect = None
            self.x1Vect = None
            self.v0Vect = None
            self.v1Vect = None
            self.dVect = None
            self.ndof = 0
            self.switchpointsList = []
            self.duration = 0.0
        else:
            # Check first if the input is valid
            minDur = curves[0].duration
            for curve in curves[1:]:
                assert( FuzzyEquals(curve.duration, minDur, epsilon) )
                minDur = min(curve.duration, minDur)

            self.curves = deepcopy(curves)
            self.duration = minDur
            self.ndof = len(self.curves)
            self.x0Vect = np.asarray([curve.x0 for curve in self.curves])
            self.x1Vect = np.asarray([curve.x1 for curve in self.curves])
            self.v0Vect = np.asarray([curve.v0 for curve in self.curves])
            self.v1Vect = np.asarray([curve.v1 for curve in self.curves])
            self.dVect = np.asarray([curve.d for curve in self.curves])

            allSwitchPointsList = deepcopy(curves[0].switchpointsList)
            for curve in self.curves[1:]:
                allSwitchPointsList += curve.switchpointsList
            allSwitchPointsList.sort() # a sorted list of all switch points (including deplicate)

            self.switchpointsList = [] # a sorted list of all distinct switch points
            if len(allSwitchPointsList) > 0:
                self.switchpointsList.append(allSwitchPointsList[0])
                for sw in allSwitchPointsList[1:]:
                    if not FuzzyEquals(sw, self.switchpointsList[-1], epsilon):
                        self.switchpointsList.append(sw)


    def Initialize(self, curves):
        if len(curves) == 0:
            self.curves = []
            self.x0Vect = None
            self.x1Vect = None
            self.v0Vect = None
            self.v1Vect = None
            self.dVect = None
            self.ndof = 0
            self.switchpointsList = []
            self.duration = 0.0
        else:
            # Check first if the input is valid
            minDur = curves[0].duration
            for curve in curves[1:]:
                assert( FuzzyEquals(curve.duration, minDur, epsilon) )
                minDur = min(curve.duration, minDur)

            self.curves = deepcopy(curves)
            self.duration = minDur
            self.ndof = len(self.curves)
            self.x0Vect = np.asarray([curve.x0 for curve in self.curves])
            self.x1Vect = np.asarray([curve.x1 for curve in self.curves])
            self.v0Vect = np.asarray([curve.v0 for curve in self.curves])
            self.v1Vect = np.asarray([curve.v1 for curve in self.curves])
            self.dVect = np.asarray([curve.d for curve in self.curves])

            allSwitchPointsList = deepcopy(curves[0].switchpointsList)
            for curve in self.curves[1:]:
                allSwitchPointsList += curve.switchpointsList
            allSwitchPointsList.sort() # a sorted list of all switch points (including deplicate)

            self.switchpointsList = [] # a sorted list of all distinct switch points
            if len(allSwitchPointsList) > 0:
                self.switchpointsList.append(allSwitchPointsList[0])
                for sw in allSwitchPointsList[1:]:
                    if not FuzzyEquals(sw, self.switchpointsList[-1], epsilon):
                        self.switchpointsList.append(sw)
            
    
    def __getitem__(self, index):
        return self.curves[index]


    def __len__(self):
        return len(self.curves)


    def IsEmpty(self):
        return len(self) == 0
    
    
    def SetInitialValue(self, x0Vect):
        self.x0Vect = np.array(x0Vect) # make a copy
        for (i, curve) in enumerate(self.curves):
            curve.SetInitialValue(self.x0Vect[i])
        self.x1Vect = np.asarray([x0 + d for (x0, d) in zip(self.x0Vect, self.dVect)])


    def EvalPos(self, t):
        assert(t >= -epsilon)
        assert(t <= self.duration + epsilon)
        
        xVect = [curve.EvalPos(t) for curve in self.curves]
        return np.asarray(xVect)


    def EvalVel(self, t):
        assert(t >= -epsilon)
        assert(t <= self.duration + epsilon)
        
        return np.asarray([curve.EvalVel(t) for curve in self.curves])


    def EvalAcc(self, t):
        assert(t >= -epsilon)
        assert(t <= self.duration + epsilon)
        
        return np.asarray([curve.EvalAcc(t) for curve in self.curves])

    
    def SetConstant(self, x0Vect, t):        
        ndof = len(x0Vect)
        curves = []
        for i in xrange(ndof):
            curve = ParabolicCurve()
            curve.SetConstant(x0Vect[i], t)
            curves.append(curve)

        self.Initialize(curves)
        return


    def SetSegment(self, x0Vect, x1Vect, v0Vect, v1Vect, t):
        assert(t >= 0)

        ndof = len(x0Vect)
        curves = []
        for i in xrange(ndof):
            curve = ParabolicCurve()
            curve.SetSegment(x0Vect[i], x1Vect[i], v0Vect[i], v1Vect[i], t)
            curves.append(curve)

        self.Initialize(curves)
        return

    
    def SetZeroDuration(self, x0Vect, v0Vect):
        ndof = len(x0Vect)
        curves = []
        for i in xrange(ndof):
            curve = ParabolicCurve()
            curve.SetZeroDuration(x0Vect[i], v0Vect[i])
            curves.append(curve)

        self.Initialize(curves)
        return


    def Cut(self, t):
        if (t <= 0):
            remCurvesND = ParabolicCurvesND()
            remCurvesND.SetZeroDuration(self.x0Vect, self.v0Vect)
            return
        elif (t >= self.duration):
            remCurvesND = ParabolicCurvesND()
            remCurvesND.SetZeroDuration(self.x1Vect, self.v1Vect)
            return

        leftHalf = self.curves
        rightHalf = []
        for i in xrange(self.ndof):
            rightHalf.append(leftHalf[i].Cut(t))

        self.Initialize(leftHalf)
        remCurvesND = ParabolicCurvesND(rightHalf)
        return remCurvesND


    def TrimFront(self, t):
        if (t <= 0):
            return
        elif (t >= self.duration):
            self.SetZeroDuration(self.x1Vect, self.v1Vect)
            return

        newCurves = self.curves
        for i in xrange(self.ndof):
            newCurves[i].TrimFront(t)
        self.Initialize(newCurves)
        return


    def TrimBack(self, t):
        if (t <= 0):
            self.SetZeroDuration(self.x0Vect, self.v0Vect)
            return
        elif (t >= self.duration):
            return

        newCurves = self.curves
        for i in xrange(self.ndof):
            newCurves[i].TrimBack(t)
        self.Initialize(newCurves)
        return


    # Visualization
    def PlotPos(self, fignum='Displacement Profiles', includingSW=False, dt=0.005):
        """Plot trajectory position vs time.

        Parameters
        ----------
        fignum : int, float, string, optional
            Figure's number/title
        includingSW : bool, optional
            If True, draw vertical straight lines at switch points.
        dt : float, optional
            Time resolution to evaluate the position.
        
        """
        plt.figure(fignum)

        tVect = np.arange(0, self.duration, dt)
        if tVect[-1] < self.duration:
            tVect = np.append(tVect, self.duration)

        xVect = [self.EvalPos(t) for t in tVect]
        plt.plot(tVect, xVect, linewidth=2)
        handle = ['joint {0}'.format(i + 1) for i in xrange(self.ndof)]
        plt.legend(handle)

        if includingSW:
            ax = plt.gca().axis()
            for s in self.switchpointsList:
                plt.plot([s, s], [ax[2], ax[3]], 'r', linewidth=1)
        plt.show(False)
        

    def PlotVel(self, fignum='Velocity Profiles', includingSW=False, **kwargs):
        """Plot trajectory velocity vs time.

        Parameters
        ----------
        fignum : int, float, string, optional
            Figure's number/title
        includingSW : bool, optional
            If True, draw vertical straight lines at switch points.
        
        """
        plt.figure(fignum)
        plt.hold(True)

        lines = []
        for curve in self.curves:
            lines.append(curve.PlotVel(fignum=fignum, **kwargs))

        handles = ['joint {0}'.format(i + 1) for i in xrange(self.ndof)]
        plt.legend(lines, handles)

        if includingSW:
            ax = plt.gca().axis()
            for s in self.switchpointsList:
                plt.plot([s, s], [ax[2], ax[3]], 'r', linewidth=1)
        plt.show(False)
        

    def PlotAcc(self, fignum='Acceleration Profiles', **kwargs):
        """Plot trajectory acceleration vs time.

        Parameters
        ----------
        fignum : int, float, string, optional
            Figure's number/title
        
        """
        plt.figure(fignum)
        plt.hold(True)

        lines = []
        for curve in self.curves:
            lines.append(curve.PlotAcc(fignum=fignum, **kwargs))

        handles = ['joint {0}'.format(i + 1) for i in xrange(self.ndof)]
        plt.legend(lines, handles)
        plt.show(False)

            
