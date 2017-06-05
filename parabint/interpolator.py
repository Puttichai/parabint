import numpy as np
import bisect
from .trajectory import Ramp, ParabolicCurve, ParabolicCurvesND
from .utilities import *
from .checker import *
import random
_rng = random.SystemRandom()

import logging
logging.basicConfig(format='[%(levelname)s] [%(name)s: %(funcName)s] %(message)s',
                    level=logging.DEBUG)
log = logging.getLogger(__name__)

_defaultGridRes = 50
_gridThreshold = 8 # if there are more than _gridThreshold grid lines, try PLP first
_defaultMinSwitch = 8e-3

# Math operations
Sqrt = np.sqrt


#
# ND Trajectory
#
def ComputeZeroVelNDTrajjectory(x0Vect, x1Vect, vmVect, amVect, delta=None):
    ndof = len(x0Vect)
    assert(ndof == len(x1Vect))
    assert(ndof == len(vmVect))
    assert(ndof == len(amVect))

    dVect = x1Vect - x0Vect

    vMin = inf
    aMin = inf
    for idof in xrange(ndof):
        if not FuzzyZero(dVect[idof], epsilon):
            vMin = min(vMin, vmVect[idof]/dVect[idof])
            aMin = min(aMin, amVect[idof]/dVect[idof])

    if not ((vMin < inf) and (aMin < inf)):
        curvesnd = ParabolicCurvesND()
        curvesnd.SetConstant(x0Vect, 0.0)
        return curvesnd

    if delta is None:
        sdProfile = _Compute1DTrajectoryWithoutDelta(0.0, 1.0, 0.0, 0.0, vMin, aMin)
    else:
        sdProfile = ComputeZeroVel1DTrajectoryWithDelta(0.0, 1.0, vMin, aMin, delta)


    curves = [ParabolicCurve() for _ in xrange(ndof)]
    for sdRamp in sdProfile:
        aVect = sdRamp.a * dVect
        v0Vect = sdRamp.v0 * dVect
        dur = sdRamp.duration

        for idof in xrange(ndof):
            ramp = Ramp(v0Vect[idof], aVect[idof], dur)
            curve = ParabolicCurve([ramp])
            curves[idof].Append(curve)

    for (i, curve) in enumerate(curves):
        curve.SetInitialValue(x0Vect[i])
    curvesnd = ParabolicCurvesND(curves)

    # Check before returning
    return curvesnd


def ComputeArbitraryVelNDTrajectory(x0Vect, x1Vect, v0Vect, v1Vect, xminVect, xmaxVect,
                                    vmVect, amVect, delta=None, tryHarder=False):
    ndof = len(x0Vect)
    assert(ndof == len(x1Vect))
    assert(ndof == len(v0Vect))
    assert(ndof == len(v1Vect))
    assert(ndof == len(xminVect))
    assert(ndof == len(xmaxVect))
    assert(ndof == len(vmVect))
    assert(ndof == len(amVect))

    dVect = x1Vect - x0Vect

    curves = []
    maxDuration = 0.0
    maxIndex = 0
    for idof in xrange(ndof):
        if delta is None:
            curve = _Compute1DTrajectoryWithoutDelta(x0Vect[idof], x1Vect[idof],
                                                     v0Vect[idof], v1Vect[idof],
                                                     vmVect[idof], amVect[idof])
        else:
            curve = _Compute1DTrajectoryWithDelta(x0Vect[idof], x1Vect[idof],
                                                  v0Vect[idof], v1Vect[idof],
                                                  vmVect[idof], amVect[idof], delta)
        curves.append(curve)
        if curve.duration > maxDuration:
            maxDuration = curve.duration
            maxIndex = idof

    log.debug("maxIndex = {0}".format(maxIndex))
    if delta == 0.0:
        stretchedCurves = _RecomputeNDTrajectoryFixedDuration(curves, vmVect, amVect,
                                                              maxIndex, tryHarder)
    else:
        stretchedCurves = _RecomputeNDTrajectoryFixedDurationWithDelta(curves, vmVect, amVect,
                                                                       maxIndex, delta, tryHarder)

    if len(stretchedCurves) == 0:
        return ParabolicCurvesND()
    
    newCurves = []
    for (i, curve) in enumerate(stretchedCurves):
        newCurve = ImposeJointLimitFixedDuration(curve, xminVect[i], xmaxVect[i],
                                                 vmVect[i], amVect[i], delta=0.0)
        if newCurve.IsEmpty():
            return ParabolicCurvesND()
        newCurves.append(newCurve)

    # Check before returning
    return ParabolicCurvesND(newCurves)


def _RecomputeNDTrajectoryFixedDuration(curves, vmVect, amVect, maxIndex, tryHarder=False):
    ndof = len(curves)
    assert(ndof == len(vmVect))
    assert(ndof == len(amVect))
    assert(maxIndex < ndof)

    newDuration = curve[maxIndex].duration
    isPrevDurationSafe = True
    if (tryHarder):
        for idof in xrange(ndof):
            tBound = CalculateLeastUpperBoundInoperavtiveTimeInterval
            (curves[idof].x0, curves[idof].x1, curves[idof].v0, curves[idof].v1,
             vmVect[idof], amVect[idof])
            if tBound > newDuration:
                newDuration = tBound
                isPrevDurationSafe = False

    newCurves = []
    for idof in xrange(ndof):
        if (isPrevDurationSafe and idof == maxIndex):
            log.debug("joint {0} is already the slowest DOF, continue to the next DOF (if any)".format(idof))
            continue

        stretchedCurve = _Stretch1DTrajectory(curve, newDuration, vmVect[idof], amVect[idof])
        if stretchedCurve.IsEmpty():
            log.debug()
            return []
        newCurves.append(stretchedCurve)

    assert(len(newCurves) == ndof)
    return newCurves


def ComputeNDTrajectoryFixedDuration(x0Vect, x1Vect, v0Vect, v1Vect, duration,
                                     xminVect, xmaxVect, vmVect, amVect):
    assert(duration > 0)

    ndof = len(x0Vect)
    assert(ndof == len(x1Vect))
    assert(ndof == len(v0Vect))
    assert(ndof == len(v1Vect))
    assert(ndof == len(xminVect))
    assert(ndof == len(xmaxVect))
    assert(ndof == len(vmVect))
    assert(ndof == len(amVect))

    curves = []
    for idof in xrange(ndof):
        curve = Compute1DTrajectoryFixedDuration(x0Vect[idof], x1Vect[idof],
                                                 v0Vect[idof], v1Vect[idof],
                                                 duration, vmVect[idof], amVect[idof])
        if curve.IsEmpty():
            return ParabolicCurvesND()

        newCurve = ImposeJointLimitFixedDuration(curve, xminVect[idof], xmaxVect[idof],
                                                 vmVect[idof], amVect[idof])
        if newCurve.IsEmpty():
            return ParabolicCurvesND()

        curves.append(newCurve)

    # Check before returning
    return ParabolicCurvesND(curves)


#
# 1D Trajectory
#
def Compute1DTrajectory(x0, x1, v0, v1, vm, am, delta=None):
    if delta is None:
        return _Compute1DTrajectoryWithoutDelta(x0, x1, v0, v1, vm, am)
    assert(delta > 0)
    return _Compute1DTrajectoryWithDelta(x0, x1, v0, v1, vm, am, delta)


def _Compute1DTrajectoryWithoutDelta(x0, x1, v0, v1, vm, am):
    d = x1 - x0
    dv = v1 - v0
    v0Sqr = v0*v0
    v1Sqr = v1*v1
    dvSqr = v1Sqr - v0Sqr

    # Calculate the displacement caused when maximally accelerate/decelerate from v0 to v1
    if (dv == 0):
        if (d == 0):
            ramp = Ramp(0, 0, 0, x0)
            return ParabolicCurve([ramp])
        else:
            dStraight = 0.0
    elif (dv > 0):
        dStraight = 0.5*dvSqr/am
    else:
        dStraight = -0.5*dvSqr/am

    if FuzzyEquals(d, dStraight, epsilon):
        # v1 can be reached from v0 by the acceleration am or -am
        a = am if dv > 0 else -am
        ramp = Ramp(x0, a, dv/a, x0)
        return ParabolicCurve([ramp])


    sumVSqr = v0Sqr + v1Sqr
    noViolation = True
    if (d > dStraight):
        # The acceleration of the first ramp is positive
        a0 = am
        vp = Sqrt(0.5*sumVSqr + a0*d)
        if (vp > vm + epsilon):
            noViolation = False
    else:
        # The acceleration of the first ramp is negative
        a0 = -am
        vp = -Sqrt(0.5*sumVSqr + a0*d)
        if (-vp > vm + epsilon):
            noViolation = False

    a0Inv = 1.0/a0
    if noViolation:
        ramp0 = Ramp(v0, a0, (vp - v0)*a0Inv, x0)
        ramp1 = Ramp(ramp0.v1, -a0, (vp - v1)*a0Inv)
        curve = ParabolicCurve([ramp0, ramp1])
    else:
        ramps = []
        h = abs(vp) - vm
        t = h*abs(a0Inv)
        if not FuzzyEquals(abs(v0), vm, epsilon):
            ramp0 = Ramp(v0, a0, (vp - v0)*a0Inv - t, x0)
            ramps.append(ramp0)
        nom = h*h
        denom = abs(a0)*vm
        newVp = vm if vp > 0 else -vm
        ramp1 = Ramp(newVp, 0.0, 2*t + nom/denom)
        ramps.append(ramp1)
        if not FuzzyEquals(abs(v1), vm, epsilon):
            ramp2 = Ramp(newVp, -a0, (vp - v1)*a0Inv - t)
            ramps.append(ramp2)
        curve = ParabolicCurve(ramps)

    # Check before returning
    return curve


def ImposeJointLimitFixedDuration(curve, xmin, xmax, vm, am, delta):
    [bmin, bmax] = curve.GetPeaks()
    if (bmin >= xmin - epsilon) and (bmax <= xmax + epsilon):
        log.debug("Input curve does not violate joint limits")
        return curve

    duration = curve.duration
    x0 = curve.x0
    x1 = curve.EvalPos(duration)
    v0 = curve.v0
    v1 = curve.v1

    bt0 = inf
    bt1 = inf
    ba0 = inf
    ba1 = inf
    bx0 = inf
    bx1 = inf
    if (v0 > zero):
        bt0 = BrakeTime(x0, v0, xmax)
        bx0 = xmax
        ba0 = BrakeAccel(x0, v0, xmax)
    elif (v0 < zero):
        bt0 = BrakeTime(x0, v0, xmin)
        bx0 = xmin
        ba0 = BrakeAccel(x0, v0, xmin)

    if (v1 < zero):
        bt1 = BrakeTime(x1, -v1, xmax)
        bx1 = xmax
        ba1 = BrakeAccel(x1, -v1, xmax)
    elif (v1 > zero):
        bt1 = BrakeTime(x1, -v1, xmin)
        bx1 = xmin
        ba1 = BrakeAccel(x1, -v1, xmin)

    newCurve = ParabolicCurve()
    if (bt0 < duration) and (abs(ba0) <= am + epsilon):
        # Case IIa
        log.debug("Case IIa")
        firstRamp = Ramp(v0, ba0, bt0, x0)
        if (abs(x1 - bx0) < (duration - bt0)*vm):
            tempCurve1 = Interpolate1D(bx0, x1, 0.0, v1, vm, am)
            if not tempCurve1.IsEmpty():
                if (duration - bt0 > tempCurve1.duration):
                    tempCurve2 = _Stretch1DTrajectory(tempCurve1, duration - bt0, vm, am)
                    if not tempCurve2.IsEmpty():
                        tempbmin, tempbmax = tempCurve2.GetPeaks()
                        if not ((tempbmin < xmin - epsilon) or (tempbmax > xmax + epsilon)):
                            log.debug("Case IIa successful")
                            newCurve = ParabolicCurve([firstRamp] + tempCurve2.ramps)
                                        

    if (bt1 < duration) and (abs(ba1) <= am + epsilon):
        # Case IIb
        log.debug("Case IIb")
        lastRamp = Ramp(0, ba1, bt1, bx1)
        if (abs(x0 - bx1) < (duration - bt1)*vm):
            tempCurve1 = Interpolate1D(x0, bx1, v0, 0.0, vm, am)
            if not tempCurve1.IsEmpty():
                if (duration - bt1 >= tempCurve1.duration):
                    tempCurve2 = _Stretch1DTrajectory(tempCurve1, duration - bt1, vm, am)
                    if not tempCurve2.IsEmpty():
                        tempbmin, tempbmax = tempCurve2.GetPeaks()
                        if not ((tempbmin < xmin - epsilon) or (tempbmax > xmax + epsilon)):
                            log.debug("Case IIb successful")
                            newCurve = ParabolicCurve(tempCurve2.ramps + [lastRamp])          
        

    if (bx0 == bx1):
        # Case III
        if (bt0 + bt1 < duration) and (max(abs(ba0), abs(ba1)) <= am + epsilon):
            log.debug("Case III")
            ramp0 = Ramp(v0, ba0, bt0, x0)
            ramp1 = Ramp(0.0, 0.0, duration - (bt0 + bt1))
            ramp2 = Ramp(0.0, ba1, bt1)
            newCurve = ParabolicCurve([ramp0, ramp1, ramp2])
    else:
        # Case IV
        if (bt0 + bt1 < duration) and (max(abs(ba0), abs(ba1)) <= am + epsilon):
            log.debug("Case IV")
            firstRamp = Ramp(v0, ba0, bt0, x0)
            lastRamp = Ramp(0.0, ba1, bt1)
            if (abs(bx0 - bx1) < (duration - (bt0 + bt1))*vm):
                tempCurve1 = Interpolate1D(bx0, bx1, 0.0, 0.0, vm, am)
                if not tempCurve1.IsEmpty():
                    if (duration - (bt0 + bt1) >= tempCurve1.duration):
                        tempCurve2 = _Stretch1DTrajectory(tempCurve1, duration - (bt0 + bt1), vm, am)
                        if not tempCurve2.IsEmpty():
                            tempbmin, tempbmax = tempCurve2.GetPeaks()
                            if not ((tempbmin < xmin - epsilon) or (tempbmax > xmax + epsilon)):
                                log.debug("Case IV successful")
                                newCurve = ParabolicCurve([firstRamp] + tempCurve2.ramps + [lastRamp])
        

    if (newCurve.isEmpty):
        log.warn("Cannot solve for a bounded trajectory")
        log.warn("x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; xmin = {4}; xmax = {5}; vm = {6}; am = {7}; duration = {8}".\
                 format(x0, x1, v0, v1, xmin, xmax, vm, am, duration))
        return newCurve

    newbmin, newbmax = newCurve.GetPeaks()
    if (newbmin < xmin + epsilon) or (newbmax > xmax + epsilon):
        log.warn("Solving finished but the trajectory still violates the bounds")
        log.warn("x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; xmin = {4}; xmax = {5}; vm = {6}; am = {7}; duration = {8}".\
                 format(x0, x1, v0, v1, xmin, xmax, vm, am, duration))
        return ParabolicCurve()

    if CheckParabolicCurve(curve, xmin, xmax, vm, am, x0, x1, v0, v1) == PCR_Normal:
        log.debug("Successfully fixed x-bound violation")
        return newCurve
    else:
        log.warn("Cannot fix x-bound violation")
        log.warn("x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; xmin = {4}; xmax = {5}; vm = {6}; am = {7}; duration = {8}".\
                 format(x0, x1, v0, v1, xmin, xmax, vm, am, duration))
        return ParabolicCurve()
    
                 
def _Stretch1DTrajectory(curve, vm, am, duration):
    return Compute1DTrajectoryFixedDuration(curve.x0, curve.x1, curve.v0, curve.v1, vm, am, duration)


def Compute1DTrajectoryFixedDuration(x0, x1, v0, v1, vm, am, duration):
    """We want to 'stretch' this velocity profile to have a new duration of
    endTime. First, try re-interpolating this profile to have two ramps. If that
    doesn't work, try modifying the profile accordingly.

    Two-ramp case: let t = endTime (the new duration that we want), a0 and a1
    the new accelerations of the profile, t0 the duration of the new first ramp.

    Starting from

       d = (v0*t0 + 0.5*a0*(t0*t0)) + ((v0 + a0*t0)*t1 + 0.5*a1*(t1*t1)),

    where d is the displacement done by this trajectory, t1 = duration - t0,
    i.e., the duration of the second ramp.  Then we can write a0 and a1 in terms
    of t0 as

       a0 = A + B/t0
       a1 = A - B/t1,

    where A = (v1 - v0)/t and B = (2d/t) - (v0 + v1). We want to get the
    velocity profile which has minimal acceleration: set the minimization
    objective to

       J(t0) = a0*a0 + a1*a1.

    We start by calculating feasible ranges of t0 due to various constraints.

    1) Acceleration constraints for the first ramp:

       -amax <= a0 <= amax.

    From this, we have

       -amax - A <= B/t0            ---   I)
       B/t0 >= amax - A.            ---  II)

    Let sum1 = -amax - A and sum2 = amax - A. We can obtain the feasible ranges
    of t0 accordingly.

    2) Acceleration constraints for the second ramp:

       -amax <= a1 <= amax.

    From this, we have

       -amax - A <= -B/(t - t0)      --- III)
       -B/(t - t0) <= amax - A.      ---  IV)

    As before, the feasible ranges of t0 can be computed accordingly.

    We will obtain an interval iX for each constraint X. Since t0 needs to
    satisfy all the four constraints plus the initial feasible range [0,
    endTime], we will obtain only one single feasible range for t0. (Proof
    sketch: intersection operation is associative and intersection of two
    intervals gives either an interval or an empty set.)

    """
    if (duration < -epsilon):
        return ParabolicCurve()

    if (duration <= epsilon):
        if FuzzyEquals(x0, x1, epsilon) and FuzzyEquals(v0, v1, epsilon):
            ramp0 = Ramp(v0, 0, 0, x0)
            curve = ParabolicCurve([ramp0])
            # Check before returning
            return curve
        else:
            log.info("newDuration is too short for any movement to be made")
            return ParabolicCurve()

    # Correct small discrepancies if any
    if (v0 > vm):
        if FuzzyEquals(v0, vm, epsilon):
            v0 = vm
        else:
            log.info("v0 > vm: {0} > {1}".format(v0, vm))
            return ParabolicCurve()
    elif (v0 < -vm):
        if FuzzyEquals(v0, -vm, epsilon):
            v0 = -vm
        else:
            log.info("v0 < -vm: {0} < {1}".format(v0, -vm))
            return ParabolicCurve()
    if (v1 > vm):
        if FuzzyEquals(v1, vm, epsilon):
            v1 = vm
        else:
            log.info("v1 > vm: {0} > {1}".format(v1, vm))
            return ParabolicCurve()
    elif (v1 < -vm):
        if FuzzyEquals(v1, -vm, epsilon):
            v1 = -vm
        else:
            log.info("v1 < -vm: {0} < {1}".format(v1, -vm))
            return ParabolicCurve()

    d = x1 - x0
    durInverse = 1.0/duration
    A = (v1 - v0)*durInverse
    B = (2*d*durInverse) - (v0 + v1)

    # A velocity profile having t = duration connecting (x0, v0) and (x1, v1)
    # will have one ramp iff    
    #        x1 - x0 = dStraight
    #              d = 0.5*(v0 + v1)*duration
    # The above equation is actually equivalent to
    #              B = 0.
    # Therefore, if B = 0 we can just interpolate the trajectory right away and return early.
    if FuzzyZero(B, epsilon):
        # giving priority to displacement and consistency between acceleration and
        # displacement
        a = 2*(x1 - x0 - v0*duration)*durInverse*durInverse; 
        ramp0 = Ramp(v0, a, duration, x0)
        curve = ParabolicCurve([ramp0])
        # Check before returning
        return curve


    sum1 = -am - A
    sum2 = am - A
    C = B/sum1
    D = B/sum2

    # Now we need to check a number of feasible intervals of tswitch1 induced by
    # constraints on the acceleration. Instead of having a class representing an
    # interval, we use the interval bounds directly. Naming convention: iXl =
    # lower bound of interval X, iXu = upper bound of interval X.
    i0l = 0
    i0u = duration
    i1l = -inf
    i1u = inf
    i2l = -inf
    i2u = inf
    i3l = -inf
    i3u = inf
    i4l = -inf
    i4u = inf
    import IPython; IPython.embed()
    if (sum1 == 0):
        if (B == 0):
            # t0 can be anything
            pass
        else:
            i1l = inf
    elif (sum1 > 0):
        log.debug("sum1 > 0. This implies that duration is too short")
        log.debug("x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; vm = {4}; am = {5}; duration = {6}".format(x0, x1, v0, v1, vm, am, duration))
        return ParabolicCurve()
    else:
        i1l = C

    if (sum2 == 0):
        if (B == 0):
            pass
        else:
            i2l = inf
    elif (sum2 > 0):
        i2l = D
    else:
        log.debug("sum2 > 0. This implies that duration is too short")
        log.debug("x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; vm = {4}; am = {5}; duration = {6}".format(x0, x1, v0, v1, vm, am, duration))
        return ParabolicCurve()

    if (i1l > i2u) or (i1u < i2l):
        log.debug("Interval 1 and interval 2 do not have any intersection")
        log.debug("x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; vm = {4}; am = {5}; duration = {6}".format(x0, x1, v0, v1, vm, am, duration))
        return ParabolicCurve()
    else:
        i2l = max(i1l, i2l)
        i2u = min(i1u, i2u)

    if (sum1 == 0):
        if (B == 0):
            # t0 can be anything
            pass
        else:
            i3l = inf
    elif (sum1 > 0):
        log.debug("sum1 > 0. This implies that duration is too short")
        log.debug("x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; vm = {4}; am = {5}; duration = {6}".format(x0, x1, v0, v1, vm, am, duration))
        return ParabolicCurve()
    else:
        i3u = duration + C

    if (sum2 == 0):
        if (B == 0):
            pass
        else:
            i4l = inf
    elif (sum2 > 0):
        i4u = duration + D
    else:
        log.debug("sum2 > 0. This implies that duration is too short")
        log.debug("x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; vm = {4}; am = {5}; duration = {6}".format(x0, x1, v0, v1, vm, am, duration))
        return ParabolicCurve()

    if (i3l > i4u) or (i3u < i4l):
        log.debug("Interval 3 and interval 4 do not have any intersection")
        log.debug("x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; vm = {4}; am = {5}; duration = {6}".format(x0, x1, v0, v1, vm, am, duration))
    else:
        i4l = max(i3l, i4l);
        i4u = min(i3u, i4u);

    if FuzzyEquals(i2l, i4u, epsilon) or FuzzyEquals(i2u, i4l, epsilon):
        log.debug("Interval 2 and interval 4 intersect at a point, most likely because the given duration is actually its minimum time")
        curve = _Compute1DTrajectoryWithoutDelta(x0, x1, v0, v1, vm, am)
        if curve.IsEmpty():
            log.debug("x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; vm = {4}; am = {5}; duration = {6}".format(x0, x1, v0, v1, vm, am, duration))
            return curve
        else:
            if FuzzyEquals(curve.duration, duration, epsilon):
                return curve
            else:
                log.debug("x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; vm = {4}; am = {5}; duration = {6}".format(x0, x1, v0, v1, vm, am, duration))
                return ParabolicCurve()
    elif (i2l > i4u) or (i2u < i4l):
        log.debug("Interval 2 and interval 4 do not have any intersection")
        log.debug("x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; vm = {4}; am = {5}; duration = {6}".format(x0, x1, v0, v1, vm, am, duration))
    else:
        i4l = max(i2l, i4l)
        i4u = min(i2u, i4u)

    if (i0l > i4u) or (i0u < i4l):
        log.debug("Interval 0 and interval 4 do not have any intersection")
        log.debug("x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; vm = {4}; am = {5}; duration = {6}".format(x0, x1, v0, v1, vm, am, duration))
        return ParabolicCurve()
    else:
        i4l = max(i0l, i4l)
        i4u = min(i0u, i4u)

    # Now we have already obtained a range of feasible values for t0 (the
    # duration of the first ramp). We choose a value of t0 by selecting the one
    # which minimize J(t0) := (a0^2 + a1^2).    
    # Let x = t0 for convenience. We can write J(x) as
    #        J(x) = (A + B/x)^2 + (A - B/(t - x))^2.
    # Then we find x which minimizes J(x) by examining the roots of dJ/dx.
    [solved, t0] = SolveForT0(A, B, duration, i4l, i4u)
    if not solved:
        # Solving dJ/dx = 0 failed. We just choose the midpoint of the feasible interval
        t0 = 0.5*(i4l + i4u)

    t1 = duration - t0
    if (t0 == 0) or (t1 == 0):
        ramp0 = Ramp(v0, A, duration, x0)
        curve = ParabolicCurve([ramp0])
        # Check before returning
        return curve

    a0 = A + B/t0
    a1 = A - B/t1
    vp = v0 + a0*t0

    # Consistency checking
    if not FuzzyEquals(vp, v1 - a1*t1, epsilon):
        log.warn("Verification failed.")
        log.warn("x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; vm = {4}; am = {5}; duration = {6}".format(x0, x1, v0, v1, vm, am, duration))
        return ParabolicCurve()

    # Check velocity bound
    if (abs(vp) <= vm + epsilon):
        # The two-ramp profile works.
        ramp0 = Ramp(v0, a0, t0, x0)
        ramp1 = Ramp(vp, a1, t1)
        curve = ParabolicCurve([ramp0, ramp1])
        # Check before returning
        return curve
    
    else:
        vmNew = vm if vp > 0 else -vm

        if FuzzyZero(a0, epsilon) or FuzzyZero(a1, epsilon):
            log.warn("Velocity limit is violated but at least one acceleration is zero: a0 = {0}, a1 = {1}".format(a0, a1))
            log.warn("x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; vm = {4}; am = {5}; duration = {6}".format(x0, x1, v0, v1, vm, am, duration))
            return ParabolicCurve()

        a0inv = 1.0/a0
        a1inv = 1.0/a1

        dv1 = vp - vmNew
        dv2 = vmNew - v0
        dv3 = vmNew - v1
        t0Trimmed = dv2*a0inv  # from vmaxNew = dx0 + a0*t0Trimmed
        t1Trimmed = -dv3*a1inv # from dx1 = vmaxNew + a1*t1Trimmed

        """Idea: we cut the excessive area above the velocity limit and paste that on
        both sides of the velocity profile. We do not divide the area and paste
        it equally on both sides. Instead, we try to minimize the sum of the new
        accelerations squared:

               minimize    a0New^2 + a1New^2.

        Let D2 be the area of the velocity profile above the velocity limit. We have

               D2 = 0.5*dt1*dv2 + 0.5*dt2*dv3.

        Using the relations

               a0New = dv2/(t0Trimmed - dt1)    and
               a1New = -dv3/(t1Trimmed - dt2)

        we finally arrive at the equation

               A2/a0New + B2/a1New = C2,

        where A2 = dv2^2, B2 = -dv3^2, and C2 = t0Trimmed*dv2 + t1Trimmed*dv3 - 2*D2.

        Let x = a0New and y = a1New for convenience, we can formulate the problem as

               minimize(x, y)    x^2 + y^2
               subject to        A2/x + B2/y = C2.

        From the above problem, we can see that the objective function is
        actually a circle while the constraint function is a hyperbola. (The
        hyperbola is centered at (A2/C2, B2/C2)). Therefore, the minimizer is
        the point where both curves touch.

        Let p = (x0, y0) be the point that the two curves touch. Then

               (slope of the hyperbola at p)*(y0/x0) = -1,

        i.e., the tangent line of the hyperbola at p and the line connecting the origin and p are
        perpendicular. Solving the above equation gives us

               x0 = (A2 + (A2*B2*B2)^(1/3))/C2.

        """
        A2 = dv2*dv2
        B2 = -dv3*dv3
        D2 = 0.5*dv1*(duration - t0Trimmed - t1Trimmed) # area of the velocity profile above the velocity limit.
        C2 = t0Trimmed*dv2 + t1Trimmed*dv3 - 2*D2
        root = (A2*B2*B2)**(1./3.)

        if FuzzyZero(C2, epsilon):
            # This means the excessive area is too large such that after we
            # paste it on both sides of the original velocity profile, the whole
            # profile becomes one-ramp with a = 0 and v = vmNew.
            log.debug("C2 == 0. Unable to fix this case.")
            log.debug("x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; vm = {4}; am = {5}; duration = {6}".format(x0, x1, v0, v1, vm, am, duration))
            return ParabolicCurve()

        C2inv = 1.0/C2
        a0 = (A2 + root)*C2inv
        if abs(a0) > am:
            if FuzzyZero(root, epsilon*epsilon):
                # The computed a0 is exceeding the bound and its corresponding
                # a1 is zero. Therefore, we cannot fix this case. This is
                # probably because the given duration is actually less than the
                # minimum duration that it can get.
                log.debug("|a0| > am and a1 == 0: Unable to fix this case since the given duration is too short.")
                return ParabolicCurve()

            a0 = am if a0 > 0 else -am
            # Recalculate the related variable
            root = C2*a0 - A2

        # Now compute a1
        # Special case: a0 == 0. Then this implies vm == dx0. Reevaluate those above equations
        # leads to a1 = B2/C2
        if abs(a0) <= epsilon:
            a0 = 0;
            a1 = B2/C2;
            if (abs(a1) > am + epsilon):
                # The computed a1 is exceeding the bound while a0 being zero. This is similar to
                # the case above when |a0| > am and a1 == 0.
                log.debug("a0 == 0 and |a1| > am: Unable to fix this case since the given duration is too short.")
                log.debug("x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; vm = {4}; am = {5}; duration = {6}".format(x0, x1, v0, v1, vm, am, duration))
                return ParabolicCurve()
            # Recalculate the related variable
            root = C2*a0 - A2;
            
        else:
            # From the hyperbola equation, we have y = B2*x/(C2*x - A2) = B2*x/root
            if FuzzyZero(root, epsilon*epsilon):
                # Special case: a1 == 0. This implies vm == dx1. If we calculate back the value of a0,
                # we will get a0 = A2/C2 which is actually root = 0.
                a1 = 0
                a0 = A2/C2
            else:
                a1 = B2*a0/root
                if abs(a1) > am:
                    # a1 exceeds the bound, try making it stays at the bound.
                    a1 = am if a1 > 0 else -am
                    # Recalculate the related variable
                    if (C2*a1 - B2 == 0):
                        # this case means a0 == 0 which shuold have been catched from above
                        log.debug("(C2*a1 - B2 == 0) a0 shuold have been zero but a0 = %.15e", a0)
                        log.debug("x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; vm = {4}; am = {5}; duration = {6}".format(x0, x1, v0, v1, vm, am, duration))
                        return ParabolicCurve()
                    a0 = A2*a1/(C2*a1 - B2)

        # Final check on the accelerations
        if (abs(a0) > am + epsilon) or (abs(a1) > am + epsilon):
            log.debug("Cannot fix acceleration bound violation")
            log.debug("x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; vm = {4}; am = {5}; duration = {6}".format(x0, x1, v0, v1, vm, am, duration))
            return ParabolicCurve()

        if FuzzyZero(a0, epsilon) and FuzzyZero(a1, epsilon):
            log.debug("Both accelerations are zero")
            log.debug("x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; vm = {4}; am = {5}; duration = {6}".format(x0, x1, v0, v1, vm, am, duration))
            return ParabolicCurve()

        if FuzzyZero(a0, epsilon):
            t0 = duration + dv3/a1
            t1 = duration - t0
            vp = vmNew

            ramp0 = Ramp(v0, a0, t0, x0)
            ramp1 = Ramp(vp, a1, t1)
            curve = ParabolicCurve([ramp0, ramp1])
        elif FuzzyZero(a1, epsilon):
            t0 = dv2/a0
            t1 = duration - t0
            vp = vmNew

            ramp0 = Ramp(v0, a0, t0, x0)
            ramp1 = Ramp(vp, a1, t1)
            curve = ParabolicCurve([ramp0, ramp1])
        else:
            t0 = dv2/a0
            if (t0 < 0):
                log.debug("t0 < 0. The given duration is not achievable with the given bounds")
                log.debug("x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; vm = {4}; am = {5}; duration = {6}".format(x0, x1, v0, v1, vm, am, duration))
                return ParabolicCurve()

            vp = vmNew
            tLastRamp = -dv3/a1
            if (tLastRamp < 0):
                log.debug("tLastRamp < 0. The given duration is not achievable with the given bounds");
                log.debug("x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; vm = {4}; am = {5}; duration = {6}".format(x0, x1, v0, v1, vm, am, duration))
                return ParabolicCurve()

            if (t0 + tLastRamp > duration):
                # Final fix
                if (A == 0):
                    log.debug("(Final fix) A == 0. Cannot fix this case")
                    log.debug("x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; vm = {4}; am = {5}; duration = {6}".format(x0, x1, v0, v1, vm, am, duration))
                    return ParabolicCurve()
                t0 = (dv2 - B)/A
                if (t0 < 0):
                    log.debug("(Final fix) t0 < 0. Cannot fix this case")
                    log.debug("x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; vm = {4}; am = {5}; duration = {6}".format(x0, x1, v0, v1, vm, am, duration))
                    return ParabolicCurve()

                t1 = duration - t0
                a0 = A + (B/t0)
                a1 = A - (B/t1)

                ramp0 = Ramp(v0, a0, t0, x0)
                ramp1 = Ramp(vp, a1, t1)
                curve = ParabolicCurve([ramp0, ramp1])
            else:
                tMiddle = duration - (t0 + tLastRamp)
                if FuzzyZero(tMiddle, epsilon):
                    # The middle ramp is too short. If we leave it like this, it may cause errors later on.
                    t0 = (2*d - (v1 + vmNew)*duration)/(v0 - v1)
                    t1 = duration - t0
                    vp = vmNew
                    a0 = dv2/t0
                    a1 = -dv3/t1
                    if (abs(a0) > am + epsilon) or (abs(a1) > am + epsilon):
                        log.debug("Cannot merge into two-ramp because of acceleration limits");
                        log.debug("x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; vm = {4}; am = {5}; duration = {6}".format(x0, x1, v0, v1, vm, am, duration))
                        return ParabolicCurve()

                    ramp0 = Ramp(v0, a0, t0, x0);
                    ramp1 = Ramp(vp, a1, t1);
                    curve = ParabolicCurve([ramp0, ramp1])
                else:
                    # Three-ramp profile really works now
                    ramp0 = Ramp(v0, a0, t0, x0)
                    ramp1 = Ramp(vp, 0, tMiddle)
                    ramp2 = Ramp(vp, a1, tLastRamp)
                    curve = ParabolicCurve([ramp0, ramp1, ramp2])

        # Check before returning
        return curve
    

#
# Utilities
#
def CalculateLeastUpperBoundInoperavtiveTimeInterval(x0, x1, v0, v1, vm, am):
    """Let t be the total duration of the velocity profile, a0 and a1 be the
    accelerations of both ramps. We write, in the way that has already been
    described in SolveMinAccel, a0 and a1 in terms of t0 as

           a0 = A + B/t0        and
           a1 = A - B/(t - t0).

    Imposing the acceleration bounds, we have the following inequalities:

       from -am <= a0 <= am, we have

                     t0*sum1 <= B
                           B <= sum2*t0

       from -am <= a1 <= am, we have

               (t - t0)*sum1 <= -B
                          -B <= sum2*(t - t0),

    where sum1 = -am - A, sum2 = am - A.

    From those inequalities, we can deduce that a feasible value of t0 must fall
    in the intersection of

           [B/sum1, t + B/sum1]    and
           [B/sum2, t + B/sum2].

    Therefore, the total duration t must satisfy

           t >= B*(1/sum2 - 1/sum1)    and
           t >= B*(1/sum1 - 1/sum2).

    By substituting A = (v1 - v0)/t and B = 2*d/t - (v0 + v1) into the above
    inequalities, we have

           t >= (2*am*((2*d)/t - (v0 + v1)))/(am*am - ((v1 - v0)/t)**2)    and
           t >= -(2*am*((2*d)/t - (v0 + v1)))/(am*am - ((v1 - v0)/t)**2),

    (the inequalities are derived using Sympy). Finally, we have two solutions
    (for the total time) from each inequality. Then we select the maximum one.

    Important note: position limits are not taken into account here. The
    calculated upper bound may be invalidated because of position constraints.

    """
    d = x1 - x0

    amInv = 1.0/am
    firstTerm = (v0 + v1)*amInv

    temp1 = 2*(-am*am)*(2*am*d - v0*v0 - v1*v1)
    secondTerm1 = Sqrt(temp1)*amInv*amInv
    if (temp1 < 0):
        T0 = -1
        T1 = -1

    else:
        T0 = firstTerm + secondTerm1
        T1 = firstTerm - secondTerm1

    T1 = max(T0, T1)

    temp2 = 2*(am*am)*(2*am*d + v0*v0 + v1*v1)
    secondTerm2 = Sqrt(temp2)*amInv*amInv
    if (temp2 < 0):
        T2 = -1
        T3 = -1

    else:
        T2 = -firstTerm + secondTerm2
        T3 = -firstTerm - secondTerm2

    T3 = max(T2, T3)

    t = max(T1, T3)
    
    if (t > epsilon):
        # dStraight is the displacement produced if we were to travel with only
        # one acceleration from v0 to v1 in t. It is used to determine which
        # direction we should aceelerate first (posititve or negative
        # acceleration).
        dStraight = 0.5*(v0 + v1)*t
        amNew = am if d - dStraight > 0 else -am
        vmNew = vm if d - dStraight > 0 else -vm

        vp = 0.5*(amNew*t + v0 + v1) # the peak velocity
        if (Abs(vp) > vm):
            dExcess = (vp - vmNew)*(vp - vmNew)*amInv
            deltaTime = dExcess/vm
            t += deltaTime # the time increased from correcting the velocity bound violation
                
        # Should be no problem now.
        t = t * 1.01 # for safety reasons, we don't make t too close to the bound
        return [True, t]

    else:
        if FuzzyEquals(x1, x0, epsilon) and FuzzyZero(v0, epsilon) and FuzzyZero(v1, epsilon):
            t = 0
            return [True, t]
        else:
            log.debug("Unable to calculate the least upper bound: T0 = {0}; T1 = {1}; T2 = {2}; T3 = {3}".format(T0, T1, T2, T3))
            return [False, t]


def SolveForT0(A, B, t, l, u):
    """
    Let x = t0 for convenience. The two accelerations can be written in terms of x as

       a0 = A + B/x    and
       a1 = A - B/(t - x),

    where t is the total duration. We want to solve the following optimization problem:

       minimize(x)    J(x) = a0^2 + a1^2.

    We find the minimizer by solving dJ/dx = 0. From

       J(x) = (A + B/x)^2 + (A - B/(t - x))^2,

    we have

       dJ/dx = (2*A)*x^4 + (2*B - 4*A*t)*x^3 + (3*A*t^2 - 3*B*t)*x^2 + (A*t^3 + 3*t^2)*x + (B*t^3).
    """
    if (l < 0):
        if (u < 0):
            log.debug("The given interval is invalid: l = {0}; u = {1}".format(l, u))
            return [False, -1]
        log.debug("Invalid lower bound is given. Reset to zero.")
        l = 0

    if FuzzyZero(A, epsilon) and FuzzyZero(B, epsilon):
        if (l > 0):
            return [False, -1]
        else:
            t0 = 0
            return [True, t0]

    tSqr = t*t
    tCube = tSqr*t
    if FuzzyZero(A, epsilon):
        coeffs = np.array([2*B, -3*B*t, 3*B*tSqr, -B*tCube])
    else:
        coeffs = np.array([2*A, -4*A*t + 2*B, 3*A*tSqr - 3*B*t, -A*tCube + 3*B*tSqr, -B*tCube])

    roots = np.roots(coeffs)
    if len(roots) == 0:
        return [False, -1]

    # Find the solution which minimizes the objective function
    J = inf
    bestT = -1.0
    for root in roots:
        if (root <= u) and (root >= l):
            if FuzzyZero(root, epsilon):
                firstTerm = 0
            else:
                firstTerm = A + B/root

            if FuzzyZero(t - root, epsilon):
                secondTerm = 0
            else:
                secondTerm = A - B/(t - root)

            curObj = firstTerm*firstTerm + secondTerm*secondTerm
            if (curObj < J):
                J = curObj
                bestT = root

    if bestT < 0:
        return [False, -1]
    else:
        return [True, bestT]


def Recompute1DTrajectoryTwoRamps(curve, t0, t1):
    """Recompute a trajectory interpolating (curve.x0, curve.v0) and (curve.x1,
    curve.v1) such that the trajectory has two ramps with durations t0 and t1
    respectively.

    Given t0 and t1, there is a unique solution to this problem.

    """
    assert(t0 > 0)
    assert(t1 > 0)
    x0 = curve.x0
    d = curve.d
    v0 = curve.v0
    v1 = curve.v1

    alpha = t0*(0.5*t0 + t1)
    beta = 0.5*t1*t1
    gamma = d - v0*(t0 + t1)
    det = alpha*t1 - beta*t0 # det is provably strictly positive
    detInv = 1.0/det
    
    a0New = (gamma*t1 - beta*(v1 - v0))*detInv
    a1new = (-gamma*t0 + alpha*(v1 - v0))*detInv

    ramp0 = Ramp(v0, a0New, t0, x0)
    ramp1 = Ramp(ramp0.v1, a1New, t1)
    return ParabolicCurve([ramp0, ramp1])


def Recompute1DTrajectoryThreeRamps(curve, t0, t1, t2, vm, am):
    """Recompute a trajectory interpolating (curve.x0, curve.v0) and (curve.x1,
    curve.v1) such that the trajectory has three ramps with durations t0, t1,
    and t2 respectively.

    """
    assert(t0 > 0)
    assert(t1 > 0)
    assert(t2 > 0)
    x0 = curve.x0
    d = curve.d
    v0 = curve.v0
    v1 = curve.v1
    
    alpha = t0*(0.5*t0 + t1 + t2)
    beta = t1*(0.5*t1 + t2)
    sigma = 0.5*t2*t2
    gamma = d - v0*(t0 + t1 + t2)
    kappa = v1 - v0

    A = np.array([[alpha, beta, sigma], [t0, t1, t2]])
    b = np.array([[gamma], [kappa]])
    AAT = np.dot(A, A.T)
    pseudoinvA = np.dot(A.T, np.linalg.inv(AAT))
    xp = np.dot(pseudoinvA, b) # particular solution
    xh = np.array([[(beta*t2 - sigma*t1)/(alpha*t1 - beta*t0)],
                   [-(alpha*t2 - sigma*t0)/(alpha*t1 - beta*t0)],
                   [1.0]]) # homogenous solution

    # Solutions to Ax = b are in the form xp + k*xh. Now we need to compute a valid interval of k.
    
    l = np.array([[(max((-vm - v0)/t0, -am))], [-am], [max((-vm + v1)/t2, -am)]])
    u = np.array([[min((vm - v0)/t0, am)], [am], [min((vm + v1)/t2, am)]])

    [result0, kl0, ku0] = SolveBoundedInEq(xh[0], xp[0], l[0], u[0])
    [result1, kl1, ku1] = SolveBoundedInEq(xh[1], xp[1], l[1], u[1])
    [result2, kl2, ku2] = SolveBoundedInEq(xh[2], xp[2], l[2], u[2])
    assert(result0 and result1 and result2)

    if (kl0 > ku1) or (ku0 < kl1):
        return ParabolicCurve()

    kl0 = max(kl0, kl1)
    ku0 = min(ku0, ku1)
    if (kl0 > ku2) or (ku0 < kl2):
        return ParabolicCurve()

    kl = max(kl0, kl2)
    ku = min(ku0, ku2)

    # Now we can choose any value k \in [kl, ku]. If there is no other
    # preference, we just randomly choose one.
    k = _rng.uniform(kl, ku)
    x = xp + k*xh

    [a0, a1, a2] = x
    ramp0 = Ramp(v0, a0, t0, x0)
    ramp1 = Ramp(ramp0.v1, a1, t1)
    ramp2 = Ramp(ramp1.v1, a2, t2)
    return ParabolicCurve([ramp0, ramp1, ramp2])
    
    
################################################################################

#
# ND Trajectory with minimum-switch-time constraint
#
def _RecomputeNDTrajectoryFixedDurationWithDelta(curves, vmVect, amVect, maxIndex, delta):
    ndof = len(curves)
    tmax = curves[maxIndex].duration

    grid = _ComputeGrid(curves[maxIndex], delta)
    PLPFirst = len(grid) > _gridThreshold
    if PLPFirst:
        first = SnapToGrid_ThreeRamps
        second = SnapToGrid_TwoRamps
    else:
        first = SnapToGrid_TwoRamps
        second = SnapToGrid_ThreeRamps
        
    newcurves = []
    for j in xrange(ndof):
        if j == maxIndex:
            newcurves.append(curves[j])
            continue

        newcurve = first(curves[j], vmVect[j], amVect[j], delta, tmax, grid)
        if len(newcurve) == 0:
            newcurve = second(curves[j], vmVect[j], amVect[j], delta, tmax, grid)
            if len(newcurve) == 0:
                log.debug("DOF {0} is infeasible.".format(j))
                return ParabolicCurvesND()
        newcurves.append(newcurve)

    return ParabolicCurvesND(newcurves)        


def _ComputeGrid(curve, delta):
    """This function compute a grid which devides each ramp of the given curve into
    intervals of equal duration t, such that t is smallest possible but still greater
    than delta.

    Grid resolution at each ramp may be different.

    """
    totalNumGrid = sum([np.floor(r.duration/delta) for r in curve])
    if totalNumGrid < _defaultGridRes:
        grids = np.array([])
        startTime = 0
        for ramp in curve:
            n = np.floor(ramp.duration/delta)
            endTime = startTime + ramp.duration
            grid = np.linspace(startTime, endTime, num=n, endpoint=False)
            grids = np.append(grids, grid)
            startTime = endTime
        grids = np.append(grids, curve.duration)
    else:
        nums = [np.floor(_defaultGridRes*r.duration/curve.duration) for r in curve]
        grids = np.array([])
        startTime = 0
        for (n, ramp) in zip(nums, curve):
            endTime = startTime + ramp.duration
            grid = np.linspace(startTime, endTime, num=n, endpoint=False)
            grids = np.append(grids, grid)
            startTime = endTime
        grids = np.append(grids, curve.duration)

    return grids


def SnapToGrid_TwoRamps(curve, vm, am, delta, tmax, grid):
    """This function try to generate a two-ramp trajectory which has the switch
    point lying at one of the grid lines. It iterates through all available grid
    lines.

    """
    for g in grid:
        assert(g > 0)
        newcurve = Recompute1DTrajectoryTwoRamps(curve, g, tmax - g)
        if CheckParabolicCurve(curve, xmin, xmax, vm, am, x0, x1, v0, v1) == PCR_Normal:
            return newcurve
    return ParabolicCurve()


def SnapToGrid_ThreeRamps(curve, vm, am, delta, tmax, grid):
    """This function try to generate a two-ramp trajectory which has the switch
    point lying at one of the grid lines.

    Since there are too many possible combinations of the two switch points, it
    make an initial guess and then try out four possibilities.

    """
    # Initial guess
    t0 = tmax * 0.25
    t1 = tmax * 0.50

    index0 = bisect.bisect_left(grid, t0)
    # i0 is a list of positions of the first switch point
    if index0 == 0:
        i0 = [index0]
    else:
        i0 = [index0 - 1, index0]

    index1 = bisect.bisect_left(grid, t1)
    # i1 is a list of positions of the second switch point
    if index1 == len(grid):
        i1 = [index1]
    else:
        i1 = [index1 - 1, index1]

    for index0 in i0:
        for index1 in i1:
            if index1 <= index0:
                continue
            t0New = grid[index0]
            t1New = grid[index1] - grid[index0]
            t2New = tmax - grid[index1]
            newcurve = Recompute1DTrajectoryThreeRamps(curve, t0New, t1New, t2New, vm, am)
            if CheckParabolicCurve(curve, xmin, xmax, vm, am, x0, x1, v0, v1) == PCR_Normal:
                return newcurve
    return ParabolicCurve()
        

################################################################################

#
# 1D Trajectory with minimum-switch-time constraint
#
def ComputeZeroVel1DTrajectoryWithDelta(x0, x1, vm, am, delta):
    curve = _Compute1DTrajectoryWithoutDelta(x0, x1, 0.0, 0.0, vm, am)
    
    if len(curve) == 1:
        assert(not (x0 == 0 and x1 == 0))
        return curve
    elif len(curve) == 2:
        return _FixSwitchTimeZeroVelPP(curve, delta)
    else:
        # len(curve) == 3 in this case
        return _FixSwitchTimeZeroVelPLP(curve, vm, am, delta)


def _FixSwitchTimeZeroVelPP(curve, delta):
    if curve[0].duration >= delta:
        return curve

    newVp = curve.d/delta # the new peak velocity
    a0 = newVp/delta
    ramp0 = Ramp(0.0, a0, delta, curve.x0)
    ramp1 = Ramp(newVp, -a0, delta)
    return ParabolicCurve([ramp0, ramp1])


def _FixSwitchTimeZeroVelPLP(curve, vm, am, delta):
    if curve[0].duration >= delta and curve[1].duration >= delta:
        # Note that we do not have to check the last ramp since it has
        # the same duration as the first ramp
        return curve

    d = curve.d
    x0 = curve.x0

    if (am*delta <= vm):
        # two-ramp
        vp = vm if d > 0 else -vm
        t0 = d/vp
        a0 = vp/t0
        ramp0 = Ramp(0.0, a0, t0, x0)
        ramp1 = Ramp(vp, -a0, t0)
        curveA = ParabolicCurve([ramp0, ramp1])
        
        # three-ramp
        a0 = am if d > 0 else -am
        vp = 0.5*(-a0*delta + np.sign(a0)*np.sqrt((a0*delta)**2 + 4*a0*d))
        if (vp/a0 < delta):
            # (delta, delta, delta)
            vp = 0.5*d/delta
            ramp0 = Ramp(0.0, vp/delta, delta, x0 = x0)
            ramp1 = Ramp(vp, 0.0, delta)
            ramp2 = Ramp(vp, -vp/delta, delta)
        else:
            # (> delta, delta, > delta)
            ramp0 = Ramp(0.0, a0, vp/a0, x0 = x0)
            ramp1 = Ramp(vp, 0.0, delta)
            ramp2 = Ramp(vp, -a0, vp/a0)
        curveB = ParabolicCurve([ramp0, ramp1, ramp2])

        # Compare the durations
        if curveA.duration <= curveB.duration:
            return curveA
        else:
            return curveB
            
    else:
        deltaInv = 1.0/delta
        if (abs(d) <= vm*delta):
            # two-ramp (delta, delta)
            vp = d*deltaInv
            a0 = vp*deltaInv
            ramp0 = Ramp(0.0, a0, delta, x0)
            ramp1 = Ramp(vp, -a0, delta)
            return ParabolicCurve([ramp0, ramp1])
        elif (abs(d) > 2*vm*delta):
            # (delta, > delta, delta)
            vp = np.sign(d)*vm
            a0 = vp*deltaInv
            ramp0 = Ramp(0.0, a0, delta, x0)
            ramp2 = Ramp(vp, -a0, delta)
            dRem = d - (ramp0.d + ramp2.d) # the remaining distance for the middle ramp
            ramp1 = Ramp(vp, 0.0, dRem/vp)
            return ParabolicCurve([ramp0, ramp1, ramp2])
        elif (abs(d) <= 1.5*vm*delta):
            # two-ramp (>= delta, >= delta)
            vp = np.sign(d)*vm
            t = d/vp
            a0 = vp/t
            ramp0 = Ramp(0.0, a0, t, x0)
            ramp1 = Ramp(vp, -a0, t)
            return ParabolicCurve([ramp0, ramp1])
        else:
            # three-ramp (delta, delta, delta)
            vp = 0.5*d*deltaInv
            a0 = vp*deltaInv
            ramp0 = Ramp(0.0, a0, delta, x0)
            ramp1 = Ramp(vp, 0.0, delta)
            ramp2 = Ramp(vp, -a0, delta)
            return ParabolicCurve([ramp0, ramp1, ramp2])
    

def _Compute1DTrajectoryWithDelta(x0, x1, v0, v1, vm, am, delta):
    curve = _Compute1DTrajectoryWithoutDelta(x0, x1, v0, v1, vm, am)

    if len(curve) == 1:
        return _FixSwitchTimeOneRamp(curve, vm, am, delta)
    elif len(curve) == 2:
        return _FixSwitchTimeTwoRamps(curve, vm, am, delta)
    else:
        return _FixSwitchTimeThreeRamps(curve, vm, am, delta)


def _FixSwitchTimeOneRamp(curve, vm, am, delta):
    # To be determined whether the fix in this case is worth it because the duration
    # of the modified ParabolicCurve tends to be much greater than the original
    # duration.
    return ParabolicCurve()
    

def _FixSwitchTimeTwoRamps(curve, vm, am, delta):
    t0 = curve[0].duration
    t1 = curve[1].duration

    if (t0 >= delta) and (t1 >= delta):
        return curve
    elif (t0 < delta) and (t1 >= delta):
        return _PP1(curve, vm, am, delta)
    elif (t0 >= delta) and (t1 < delta):
        return _PP2(curve, vm, am, delta)
    else:
        return _PP3(curve, vm, am, delta)

        
def _PP1(curve, vm, am, delta):
    """(t0 < delta) and (t1 >= delta)

    Stretch the duration of the first ramp to delta while the second ramp remains at
    the same acceleration (hence shorter). After that there are three possible cases.
    
    A: no further correction is needed.
    B: the second ramp becomes shorter than delta and re-interpolating (x0, v0) and (x1, v1)
       using a trajectory with 2 delta-ramps is better
    C: the second ramp becomes shorter than delta and re-interpolating (x0, v0) and (x1, v1)
       using a trajectory with one ramp is better
    """
    ramp0 = curve[0]
    ramp1 = curve[-1]
    t0 = ramp0.duration
    t1 = ramp1.duration
    v0 = ramp0.v0
    v1 = ramp1.v1
    a0 = ramp0.a
    a1 = ramp1.a
    d = curve.d
    x0 = ramp0.x0
    vp = ramp0.v1
    deltaInv = 1.0/delta

    if FuzzyZero(a1, epsilon):
        # Stretch the first ramp
        ramp0A = Ramp(v0, (vp - v0)*deltaInv, delta, x0)
        dRem = d - ramp0A.d
        if FuzzyZero(vp, epsilon):
            log.warn("The peak velocity is zero")
            return ParabolicCurve()
        t1New = dRem/vp

        if (t1New >= delta):
            ramp1A = Ramp(vp, 0.0, t1New)
            curveA = ParabolicCurve([ramp0A, ramp1A])
            # Check before returning
            return curveA
    else:
        k = a1*delta
        vpNew = 0.5*(k - np.sign(a1)*np.sqrt(k**2 + 4*(k*v0 + v1**2) - 8*a1*d))
        t1New = (v1 - vpNew)/a1

        if (t1New >= delta):
            # A: no further correction is needed.
            log.debug("PP1 A")
            a0New = (vpNew - v0)*deltaInv
            ramp0A = Ramp(v0, a0New, delta, x0)
            ramp1A = Ramp(vpNew, a1, t1New)
            curveA = ParabolicCurve([ramp0A, ramp1A])
            # Check before returning
            return curveA

    # If arrive here, case A does not work. Need some modification.
    # Try case B.
    vpNew = d*deltaInv - 0.5*(v0 + v1)
    a0New = (vpNew - v0)*deltaInv
    a1New = (v1 - vpNew)*deltaInv

    if (abs(a0New) <= am + epsilon) and (abs(a1New) <= am + epsilon) and (abs(vpNew) <= vm + epsilon):
        ramp0B = Ramp(v0, a0New, delta, x0)
        ramp1B = Ramp(vpNew, a1New, delta)
        curveB = ParabolicCurve([ramp0B, ramp1B])
    else:
        # Case B does not produce any feasible solution
        curveB = ParabolicCurve()

    # Try case C.
    if FuzzyZero(v0 + v1, epsilon):
        # Case C does not produce any feasible solution
        curveC = ParabolicCurve()
    else:
        tNew = 2*d/(v0 + v1)
        aNew = (v1 - v0)/tNew
        if (abs(aNew) <= am + epsilon) and (tNew >= delta):
            ramp0C = Ramp(v0, aNew, tNew, x0)
            curveC = ParabolicCurve([ramp0C])
        else:
            # Case C does not produce any feasible solution
            curveC = ParabolicCurve()

    # There has to be at least one valid case
    assert(not (curveB.IsEmpty() and curveC.IsEmpty()))

    if curveB.IsEmpty():
        # Check before returning
        return curveC
    elif curveC.IsEmpty():
        # Check before returning
        return curveB
    elif curveB.duration <= curveC.duration:
        # Check before returning
        return curveB
    else:
        # Check before returning
        return curveC


def _PP2(curve, vm, am, delta):
    """(t0 >= delta) and (t1 < delta)

    Stretch the duration of the second ramp to delta while the first ramp remains at
    the same acceleration (hence shorter). After that there are three possible cases.
    
    A: no further correction is needed.
    B: the first ramp becomes shorter than delta and re-interpolating (x0, v0) and (x1, v1)
       using a trajectory with 2 delta-ramps is better
    C: the first ramp becomes shorter than delta and re-interpolating (x0, v0) and (x1, v1)
       using a trajectory with one ramp is better
    """
    ramp0 = curve[0]
    ramp1 = curve[-1]
    t0 = ramp0.duration
    t1 = ramp1.duration
    v0 = ramp0.v0
    v1 = ramp1.v1
    a0 = ramp0.a
    a1 = ramp1.a
    d = curve.d
    x0 = ramp0.x0
    vp = ramp0.v1
    deltaInv = 1.0/delta

    if FuzzyZero(a0, epsilon):
        # Stretch the last ramp
        ramp1A = Ramp(vp, (v1 - v0)*deltaInv, delta)
        dRem = d - ramp1A.d
        if FuzzyZero(vp, epsilon):
            log.warn("The peak velocity is zero")
            return ParabolicCurve()
        t0New = dRem/vp
        
        if (t0New >= delta):
            ramp0A = Ramp(v0, 0.0, t0New, x0)
            curveA = ParabolicCurve([ramp0A, ramp1A])
            # Check before returning
            return curveA
    else:
        k = a0*delta
        vpNew = 0.5*( - k + np.sign(a0)*np.sqrt(k**2 - 4*(k*v1 - v0**2) + 8*a0*d))
        t0New = (vpNew - v0)/a0
        
        if (t0New >= delta):
            # A: no further correction is needed.
            log.debug("PP2 A")
            a1New = (v1 - vpNew)*deltaInv
            ramp0A = Ramp(v0, a0, t0New, x0)
            ramp1A = Ramp(vpNew, a1New, delta)
            curveA = ParabolicCurve([ramp0A, ramp1A])
            # Check before returning
            return curveA

    # If arrive here, case A does not work. Need some modification.
    # Try case B.
    vpNew = d*deltaInv - 0.5*(v0 + v1)
    a0New = (vpNew - v0)*deltaInv
    a1New = (v1 - vpNew)*deltaInv

    if (abs(a0New) <= am + epsilon) and (abs(a1New) <= am + epsilon) and (abs(vpNew) <= vm + epsilon):
        ramp0B = Ramp(v0, a0New, delta, x0)
        ramp1B = Ramp(vpNew, a1New, delta)
        curveB = ParabolicCurve([ramp0B, ramp1B])
    else:
        # Case B does not produce any feasible solution
        curveB = ParabolicCurve()

    # Try case C.
    if FuzzyZero(v0 + v1, epsilon):
        # Case C does not produce any feasible solution
        curveC = ParabolicCurve()
    else:
        tNew = 2*d/(v0 + v1)
        aNew = (v1 - v0)/tNew
        if (abs(aNew) <= am + epsilon) and (tNew >= delta):
            ramp0C = Ramp(v0, aNew, tNew, x0)
            curveC = ParabolicCurve([ramp0C])
        else:
            # Case C does not produce any feasible solution
            curveC = ParabolicCurve()

    # There has to be at least one valid case
    assert(not (curveB.IsEmpty() and curveC.IsEmpty()))

    if curveB.IsEmpty():
        # Check before returning
        return curveC
    elif curveC.IsEmpty():
        # Check before returning
        return curveB
    elif curveB.duration <= curveC.duration:
        # Check before returning
        return curveB
    else:
        # Check before returning
        return curveC
    
    
def _PP3(curve, vm, am, delta):
    """(t0 < delta) and (t1 < delta)

    First we try out two possibilities.
    
    A: re-interpolating (x0, v0) and (x1, v1) using a trajectory with 2 delta-ramps
    B: re-interpolating (x0, v0) and (x1, v1) using a trajectory with one ramp

    If the above two cases do not give any feasible trajectory, we try to re-interpolate (x0, v0)
    and (x1, v1) again by using the idea of flipping ramps. Basically if the original trajectory
    goes with +am -> -am, the new trajectory would be -am -> +am.
    """
    ramp0 = curve[0]
    ramp1 = curve[-1]
    t0 = ramp0.duration
    t1 = ramp1.duration
    v0 = ramp0.v0
    v1 = ramp1.v1
    a0 = ramp0.a
    a1 = ramp1.a
    d = curve.d
    x0 = ramp0.x0
    vp = ramp0.v1
    deltaInv = 1.0/delta

    # Try case A.
    vpNew = d*deltaInv - 0.5*(v0 + v1)
    a0New = (vpNew - v0)*deltaInv
    a1New = (v1 - vpNew)*deltaInv
    if (abs(a0New) <= am + epsilon) and (abs(a1New) <= am + epsilon) and (abs(vpNew) <= vm + epsilon):
        ramp0A = Ramp(v0, a0New, delta, x0)
        ramp1A = Ramp(vpNew, a1New, delta)
        curveA = ParabolicCurve([ramp0A, ramp1A])
    else:
        # Case A does not produce any feasible solution
        curveA = ParabolicCurve()

    # Try case B.
    if FuzzyZero(v0 + v1, epsilon):
        # Case C does not produce any feasible solution
        curveB = ParabolicCurve()
    else:
        tNew = 2*d/(v0 + v1)
        aNew = (v1 - v0)/tNew
        if (abs(aNew) <= am + epsilon) and (tNew >= delta):
            ramp0B = Ramp(v0, aNew, tNew, x0)
            curveB = ParabolicCurve([ramp0B])
        else:
            # Case B does not produce any feasible solution
            curveB = ParabolicCurve()

    if not (curveA.IsEmpty() and curveB.IsEmpty()):
        # We already have a feasible solution
        if curveA.IsEmpty():
            log.debug("PP3 A")
            return curveB
        elif curveB.IsEmpty():
            log.debug("PP3 B")
            return curveA
        else:
            if curveA.duration < curveB.duration:
                log.debug("PP3 A")
                return curveA
            else:
                log.debug("PP3 B")
                return curveB

    log.debug("PP3 Flipping")
    # If arrive here, both cases A and B do not produce any feasible solution.
    # First see if the flipped ramps can go with maximum accel/decel
    if FuzzyZero(a0, epsilon):
        a0New = a1
        a1New = -a1
    elif FuzzyZero(a1, epsilon):
        a0New = -a0
        a1New = a0
    else:
        a0New = -a0
        a1New = -a1

    vpSqr = 0.5*(v0*v0 + v1*v1) + a0New*d
    if (vpSqr >= 0) and (not FuzzyZero(a0New, epsilon)) and (not FuzzyZero(a1New, epsilon)):
        # Both ramps can saturate the acceleration bound. Now figure out which value of vpNew to use
        vpNew = Sqrt(vpSqr)
        t0New = (vpNew - v0)/a0New
        t1New = (v1 - vpNew)/a1New
        if (t0New >= delta) and (t1New >= delta):
            ramp0C = Ramp(v0, a0New, t0New, x0)
            ramp1C = Ramp(vpNew, a1New, t1New)
            curveC = ParabolicCurve([ramp0C, ramp1C])
            # Check before returning
            return curveC

        # vpNew being positive does not work
        vpNew *= -1.0
        t0New = (vpNew - v0)/a0New
        t1New = (v1 - vpNew)/a1New
        if (t0New >= delta) and (t1New >= delta):
            ramp0C = Ramp(v0, a0New, t0New, x0)
            ramp1C = Ramp(vpNew, a1New, t1New)
            curveC = ParabolicCurve([ramp0C, ramp1C])
            # Check before returning
            return curveC

    # Now we know that the flipped velocity profile cannot saturate acceleration bound all the
    # time. We try to modify the (flipped) velocity profile using the idea in PP1 and PP2.
    if (t0 <= t1):
        # When flipped, we would have t0New > t1New. Therefore, we follow the procedure of PP1.
        k = a1New*delta
        discriminant = k**2 + 4*(k*v0 + v1**2) - 8*a1New*d
        if (discriminant < 0):
            # Fail to calculate vpNew following PP1
            return ParabolicCurve()

        vpNew = 0.5*(k - np.sign(a1New)*np.sqrt(discriminant))
        t1New = (v1 - vpNew)/a1New
        if (t1New >= delta):
            a0New = (vpNew - v0)*deltaInv
            ramp0A = Ramp(v0, a0New, delta, x0)
            ramp1A = Ramp(vpNew, a1New, t1New)
            curveA = ParabolicCurve([ramp0A, ramp1A])
            # Check before returning
            return curveA
    
    else:
        # (t0 > t1)
        # When flipped, we would have t0New < t1New. Therefore, we follow the procedure of PP2.
        k = a0New*delta
        discriminant = k**2 - 4*(k*v1 - v0**2) + 8*a0New*d
        discriminant = k**2 + 4*(k*v0 + v1**2) - 8*a1New*d
        if (discriminant < 0):
            # Fail to calculate vpNew following PP2
            return ParabolicCurve()

        vpNew = 0.5*( - k + np.sign(a0New)*np.sqrt(discriminant))
        t0New = (vpNew - v0)/a0New
        if (t0New >= delta):
            a1New = (v1 - vpNew)*deltaInv
            ramp0A = Ramp(v0, a0New, t0New, x0)
            ramp1A = Ramp(vpNew, a1New, delta)
            curveA = ParabolicCurve([ramp0A, ramp1A])
            # Check before returning
            return curveA

    # PP1A (or PP2A) modification does not produce any feasible trajectory.
    # Try case B.
    vpNew = d*deltaInv - 0.5*(v0 + v1)
    a0New = (vpNew - v0)*deltaInv
    a1New = (v1 - vpNew)*deltaInv

    if (abs(a0New) <= am + epsilon) and (abs(a1New) <= am + epsilon) and (abs(vpNew) <= vm + epsilon):
        ramp0B = Ramp(v0, a0New, delta, x0)
        ramp1B = Ramp(vpNew, a1New, delta)
        curveB = ParabolicCurve([ramp0B, ramp1B])
    else:
        # Case B does not produce any feasible solution
        curveB = ParabolicCurve()

    # Try case C.
    if FuzzyZero(v0 + v1, epsilon):
        # Case C does not produce any feasible solution
        curveC = ParabolicCurve()
    else:
        tNew = 2*d/(v0 + v1)
        aNew = (v1 - v0)/tNew
        if (abs(aNew) <= am + epsilon) and (tNew >= delta):
            ramp0C = Ramp(v0, aNew, tNew, x0)
            curveC = ParabolicCurve([ramp0C])
        else:
            # Case C does not produce any feasible solution
            curveC = ParabolicCurve()

    # There has to be at least one valid case
    if curveB.IsEmpty() and curveC.IsEmpty():
        return curveB # return an empty one
    elif curveB.IsEmpty():
        # Check before returning
        return curveC
    elif curveC.IsEmpty():
        # Check before returning
        return curveB
    elif curveB.duration <= curveC.duration:
        # Check before returning
        return curveB
    else:
        # Check before returning
        return curveC        
    

def _FixSwitchTimeThreeRamps(curve, vm, am, delta):
    t0 = curve[0].duration
    t1 = curve[1].duration
    t2 = curve[2].duration

    if (t0 >= delta) and (t1 >= delta) and (t2 >= delta):
        return curve
    elif (t0 < delta) and (t1 >= delta) and (t2 >= delta):
        return _PLP1(curve, vm, am, delta)
    elif (t0 >= delta) and (t1 >= delta) and (t2 < delta):
        return _PLP2(curve, vm, am, delta)
    elif (t0 >= delta) and (t1 < delta) and (t2 >=  delta):
        return _PLP3(curve, vm, am, delta)
    elif (t0 < delta) and (t1 < delta) and (t2 >= delta):
        return _PLP4(curve, vm, am, delta)
    elif (t0 >= delta) and (t1 < delta) and (t2 < delta):
        return _PLP5(curve, vm, am, delta)
    elif (t0 < delta) and (t1 >= delta) and (t2 < delta):
        return _PLP6(curve, vm, am, delta)
    elif (t0 < delta) and (t1 < delta) and (t2 < delta):
        return _PLP7(curve, vm, am, delta)


def _PLP1(curve, vm, am, delta):
    """(t0 < delta) and (t1 >= delta) and (t2 >= delta)

    We consider two possibilities.

    A: stretch the first ramp's duration to delta. Then use (x1, v1) of the first ramp together with
       the trajectory's (x1, v1) as boundaary conditions to interpolate a sub-trajectory
    B: merge the first two ramps into one
    """
    firstRamp = curve[0]
    middleRamp = curve[1]
    lastRamp = curve[2]

    t0 = firstRamp.duration
    t1 = middleRamp.duration
    t2 = lastRamp.duration
    v0 = firstRamp.v0
    v1 = lastRamp.v1
    x0 = firstRamp.x0
    d = curve.d
    d0 = firstRamp.d + middleRamp.d
    vp = middleRamp.v0 # the middle has zero acceleration
    deltaInv = 1.0/delta

    # Try case A.
    a0New = (vp - v0)*deltaInv
    ramp0A = Ramp(v0, a0New, delta, x0)
    dRem = d - ramp0A.d
    subCurveA = _Compute1DTrajectoryWithDelta(0, dRem, vp, v1, vm, am, delta)
    assert(not subCurveA.IsEmpty()) # this computation should not fail
    if len(subCurveA) == 1:
        curveA = ParabolicCurve([ramp0A, subCurveA[0]])
    else:
        # len(subCurveA) == 2 in this case
        curveA = ParabolicCurve([ramp0A, subCurveA[0], subCurveA[1]])
    
    # Try case B.
    if FuzzyZero(vp + v0, epsilon):
        curveB = ParabolicCurve()
    else:
        t0New = 2*d0/(vp + v0)
        a0New = (vp - v0)/t0New
        ramp0B = Ramp(v0, a0New, t0New, x0)
        curveB = ParabolicCurve([ramp0B, lastRamp])

    assert(not (curveA.IsEmpty() and curveB.IsEmpty()))
    if curveA.IsEmpty():
        # Check before returning
        log.debug("PLP1 B")
        return curveB
    elif curveB.IsEmpty():
        # Check before returning
        log.debug("PLP1 A")
        return curveA
    elif curveA.duration < curveB.duration:
        # Check before returning
        log.debug("PLP1 A")
        return curveA
    else:
        # Check before returning
        log.debug("PLP1 B")
        return curveB
    

def _PLP2(curve, vm, am, delta):
    """(t0 >= delta) and (t1 >= delta) and (t2 < delta)

    We consider two possibilities.

    A: stretch the last ramp's duration to delta. Then use (x0, v0) of the trajectory together with
       (x0, v0) of the last ramp as boundaary conditions to interpolate a sub-trajectory
    B: merge the last two ramps into one
    """
    firstRamp = curve[0]
    middleRamp = curve[1]
    lastRamp = curve[2]

    t0 = firstRamp.duration
    t1 = middleRamp.duration
    t2 = lastRamp.duration
    v0 = firstRamp.v0
    v1 = lastRamp.v1
    x0 = firstRamp.x0
    d = curve.d
    d1 = middleRamp.d + lastRamp.d
    vp = middleRamp.v0 # the middle has zero acceleration
    deltaInv = 1.0/delta

    # Try case A.
    a2New = (v1 - vp)*deltaInv
    ramp2A = Ramp(vp, a2New, delta)
    dRem = d - ramp2A.d
    subCurveA = _Compute1DTrajectoryWithDelta(0, dRem, v0, vp, vm, am, delta)
    assert(not subCurveA.IsEmpty()) # this computation should not fail
    if len(subCurveA) == 1:
        curveA = ParabolicCurve([subCurveA[0], ramp2A])
    else:
        # len(subCurveA) == 2 in this case
        curveA = ParabolicCurve([subCurveA[0], subCurveA[1], ramp2A])
    
    # Try case B.
    if FuzzyZero(vp + v1, epsilon):
        curveB = ParabolicCurve()
    else:
        t1New = 2*d1/(vp + v1)
        a1New = (v1 - vp)/t1New
        ramp1B = Ramp(vp, a1New, t1New)
        curveB = ParabolicCurve([firstRamp, ramp1B])

    assert(not (curveA.IsEmpty() and curveB.IsEmpty()))
    if curveA.IsEmpty():
        # Check before returning
        log.debug("PLP2 B")
        return curveB
    elif curveB.IsEmpty():
        # Check before returning
        log.debug("PLP2 A")
        return curveA
    elif curveA.duration < curveB.duration:
        # Check before returning
        log.debug("PLP2 A")
        return curveA
    else:
        # Check before returning
        log.debug("PLP2 B")
        return curveB


def _PLP3(curve, vm, am, delta):
    """(t0 >= delta) and (t1 < delta) and (t2 >= delta)
    
    There are three possibilities.

    A: stretch the duration of the middle ramp to delta. The acceleration of the middle might not be
       zero after the stretching
    B: merge any two ramps together to obtain a two-ramp velocity profile
    C: the resulting trajectory has 3 delta-ramps
    """
    firstRamp = curve[0]
    middleRamp = curve[1]
    lastRamp = curve[2]

    t0 = firstRamp.duration
    t1 = middleRamp.duration
    t2 = lastRamp.duration
    v0 = firstRamp.v0
    v1 = lastRamp.v1
    x0 = firstRamp.x0
    d = curve.d
    d1 = middleRamp.d + lastRamp.d
    vp = middleRamp.v0 # the middle has zero acceleration
    deltaInv = 1.0/delta
    
    # Try case A.
    if firstRamp.a > 0:
        am_ = am
        vm_ = am
    else:
        am_ = -am
        vm_ = -vm
    amInv = 1.0/am_
    k = 0.5*delta + v0*amInv
    h = 0.5*delta + v1*amInv
    r2 = amInv*amInv*(v0*v0 + v1*v1 + 2*d*am_ + 0.5*am_*am_*delta*delta)

    t0l_0 = max(delta, (v1 - v0))*amInv
    t0u_0 = (vm_ - v0)*amInv
    t2l_0 = -h + Sqrt(r2 - (k + t0u_0)**2)
    t2u_0 = -h + Sqrt(r2 - (k + t0l_0)**2)
    if t2l_0 > t2u_0:
        [t2l_0, t2u_0] = Swap(t2u_0, t2l_0)

    t2l_1 = max(delta, (v0 - v1)*amInv)
    t2u_1 = (vm_ - v1)*amInv
    t0l_1 = -k + Sqrt(r2 - (h + t2u_1)**2)
    t0u_1 = -k + Sqrt(r2 - (h + t2l_1)**2)
    if t0l_1 > t0u_1:
        [t0l_1, t0u_1] = Swap(t0u_1, t0l_1)

    t0l = max(t0l_0, t0l_1)
    t0u = min(t0u_0, t0u_1)
    t2l = max(t2l_0, t2l_1)
    t2u = min(t2u_0, t2u_1)
    if (t0l <= t0u) and (t2l <= t2u):
        duration0 = t0l + delta + t2u
        duration1 = t0u + delta + t2l
        if duration0 <= duration1:
            t0New = t0l
            t2New = t2u
        else:
            t0New = t0u
            t2New = t2l
        vp0New = v0 + am_*t0New
        vp1New = v1 + am_*t2New
        
        ramp0A = Ramp(v0, am_, t0New, x0)
        ramp1A = Ramp(vp0New, (vp1New - vp0New)*deltaInv, delta)
        ramp2A = Ramp(vp1New, -am_, t2New)
        curveA = ParabolicCurve([ramp0A, ramp1A, ramp2A])
    else:
        curveA = ParabolicCurve()

    # Try case B.
    d0 = firstRamp.d + middleRamp.d
    d1 = middleRamp.d + lastRamp.d
    # B1: merge the first two ramps together
    subCurveB1 = _Compute1DTrajectoryWithDelta(0, d0, v0, vp, vm, am, delta)
    if len(subCurveB1) == 1:
        curveB1 = ParabolicCurve([subCurveB1[0], lastRamp])
    else:
        # len(subCurveB1) == 2 in this case
        curveB1 = ParabolicCurve([subCurveB1[0], subCurveB1[1], lastRamp])

    # B2: merge the last two ramps together
    subCurveB2 = _Compute1DTrajectoryWithDelta(0, d1, vp, v1, vm, am, delta)
    if len(subCurveB2) == 1:
        curveB2 = ParabolicCurve([firstRamp, subCurveB2[0]])
    else:
        # len(subCurveB2) == 2 in this case
        curveB2 = ParabolicCurve([firstRamp, subCurveB2[0], subCurveB2[1]])

    if curveB1.duration <= curveB2.duration:
        curveB = curveB1
    else:
        curveB = curveB2

    # Try case C.
    sumVp = d*deltaInv - 0.5*(v0 + v1)
    sumVpL = v0 + v1 - 2*am*delta
    sumVpU = v0 + v1 + 2*am*delta
    if sumVpL <= sumVp <= sumVpU:
        vp0L = v0 - am*delta
        vp0U = v0 + am*delta
        vp1L = max(sumVp - vp0U, v1 - am*delta)
        vp1U = min(sumVp - vp0L, v1 + am*delta)
        if vp1L > vp1U:
            curveC = ParabolicCurve()
        else:
            passed = False
            maxTries = 1000
            # Randomly choose vp1 until the value is feasible
            for it in xrange(maxTries):
                vp1New = _rng.uniform(vp1L, vp1U)
                vp0New = sumVp - vp1New
                if abs(vp1New - vp0New) <= am*delta:
                    passed = True
                    break
            if not passed:
                curveC = ParabolicCurve()
            else:
                ramp0C = Ramp(v0, (vp0New - v0)*deltaInv, delta, x0)
                ramp1C = Ramp(vp0New, (vp1New - vp0New)*deltaInv, delta)
                ramp2C = Ramp(vp1New, (v1 - vp1New)*deltaInv, delta)
                curveC = ParabolicCurve([ramp0C, ramp1C, ramp2C])
            
    else:
        curveC = ParabolicCurve()

    # Compare all the cases and choose the best trajectory
    if not curveC.IsEmpty():
        if not curveB.IsEmpty():
            if curveB.duration <= curveC.duration:
                # Check before returning
                log.debug("PLP3 B")
                return curveB
            else:
                # Check before returning
                log.debug("PLP3 C")
                return curveC
        else:
            # Check before returning
            log.debug("PLP3 C")
            return curveC

    if curveA.IsEmpty():
        # Check before returning
        log.debug("PLP3 B")
        return curveB
    else:
        if curveB.IsEmpty():
            # Check before returning
            log.debug("PLP3 A")
            return curveA
        else:
            if curveB.duration <= curveA.duration:
                # Check before returning
                log.debug("PLP3 B")
                return curveB
            else:
                # Check before returning
                log.debug("PLP3 A")
                return curveA
            
            
def _PLP4(curve, vm, am, delta):
    """(t0 < delta) and (t1 < delta) and (t2 >= delta)
    
    This case is similar to PLP1 and we first follow the procedure of PLP1. However, PLP1A and PLP1B
    can both be infeasible. So we introduct PLP4C where we re-interpolate the trajectory to have
    two-ramp by using PP1 procedure.
    """
    firstRamp = curve[0]
    middleRamp = curve[1]
    lastRamp = curve[2]

    t0 = firstRamp.duration
    t1 = middleRamp.duration
    t2 = lastRamp.duration
    v0 = firstRamp.v0
    v1 = lastRamp.v1
    x0 = firstRamp.x0
    d = curve.d
    d0 = firstRamp.d + middleRamp.d
    vp = middleRamp.v0 # the middle has zero acceleration
    deltaInv = 1.0/delta

    # Try case A.
    a0New = (vp - v0)*deltaInv
    ramp0A = Ramp(v0, a0New, delta, x0)
    dRem = d - ramp0A.d
    subCurveA = _Compute1DTrajectoryWithDelta(0, dRem, vp, v1, vm, am, delta)
    assert(not subCurveA.IsEmpty()) # this computation should not fail
    if len(subCurveA) == 1:
        curveA = ParabolicCurve([ramp0A, subCurveA[0]])
    else:
        # len(subCurveA) == 2 in this case
        curveA = ParabolicCurve([ramp0A, subCurveA[0], subCurveA[1]])
    
    # Try case B.
    if FuzzyZero(vp + v0, epsilon):
        curveB = ParabolicCurve()
    else:
        t0New = 2*d0/(vp + v0)
        passed = False
        if (t0New >= delta):
            a0New = (vp - v0)/t0New
            if abs(a0New) <= am + epsilon:
                passed = True
                ramp0B = Ramp(v0, a0New, t0New, x0)
                curveB = ParabolicCurve([ramp0B, lastRamp])
        if not passed:
            curveB = ParabolicCurve()

    # Try case C.
    # Two-ramp velocity profile where the first ramp saturates the minimum-switch-time constraint
    # and the second ramp saturates the acceleration constraint.
    a2 = lastRamp.a
    k = a2*delta
    vpNew = 0.5*(k - np.sign(a2)*np.sqrt(k**2 + 4*(k*v0 + v1**2) - 8*a2*d))
    if abs(vpNew) > vm + epsilon:
        curveC = ParabolicCurve()
    else:
        t1New = (v1 - vpNew)/a2 # a2 is not zero
        if (t1New >= delta):
            # PP1A: no further correction is needed.
            a0New = (vpNew - v0)*deltaInv
            ramp0C_A = Ramp(v0, a0New, delta, x0)
            ramp1C_A = Ramp(vpNew, a2, t1New)
            curveC_A = ParabolicCurve([ramp0C_A, ramp1C_A])
            curveC = curveC_A
        else:
            # Try case PP1B.
            vpNew = d*deltaInv - 0.5*(v0 + v1)
            a0New = (vpNew - v0)*deltaInv
            a1New = (v1 - vpNew)*deltaInv

            if (abs(a0New) <= am + epsilon) and (abs(a1New) <= am + epsilon) and (abs(vpNew) <= vm + epsilon):
                ramp0C_B = Ramp(v0, a0New, delta, x0)
                ramp1C_B = Ramp(vpNew, a1New, delta)
                curveC_B = ParabolicCurve([ramp0C_B, ramp1C_B])
            else:
                # Case PP2B does not produce any feasible solution
                curveC_B = ParabolicCurve()

            # Try case PP1C.
            if FuzzyZero(v0 + v1, epsilon):
                # Case C does not produce any feasible solution
                curveC_C = ParabolicCurve()
            else:
                tNew = 2*d/(v0 + v1)
                aNew = (v1 - v0)/tNew
                if (abs(aNew) <= am + epsilon) and (tNew >= delta):
                    ramp0C_C = Ramp(v0, aNew, tNew, x0)
                    curveC_C = ParabolicCurve([ramp0C])
                else:
                    # Case C does not produce any feasible solution
                    curveC_C = ParabolicCurve()

            assert (curveC_B.IsEmpty() and curveC_C.IsEmpty())
            if curveC_B.IsEmpty():
                curveC = curveC_C
            elif curveC_C.IsEmpty():
                curveC = curveC_B
            else:
                curveC = curveC_B if curveC_B.duration < curveC_C.duration else curveC_C

    # Now compare PLP1A, PLP1B, and PLP1C
    if curveA.IsEmpty() and curveB.IsEmpty() and curveC.IsEmpty():
        assert False
        newCurve = curveA # empty curve
    elif (not curveA.IsEmpty()) and curveB.IsEmpty() and curveC.IsEmpty():
        log.debug("PLP4 A")
        newCurve = curveA
    elif curveA.IsEmpty() and (not curveB.IsEmpty()) and curveC.IsEmpty():
        log.debug("PLP4 B")
        newCurve = curveB
    elif curveA.IsEmpty() and curveB.IsEmpty() and (not curveC.IsEmpty()):
        log.debug("PLP4 C")
        newCurve = curveC
    elif curveA.IsEmpty():
        if curveB.duration <= curveC.duration:
            log.debug("PLP4 B")
            newCurve = curveB
        else:
            log.debug("PLP4 C")
            newCurve = curveC
    elif curveB.IsEmpty():
        if curveA.duration <= curveC.duration:
            log.debug("PLP4 A")
            newCurve = curveA
        else:
            log.debug("PLP4 C")
            newCurve = curveC
    elif curveC.IsEmpty():
        if curveA.duration <= curveB.duration:
            log.debug("PLP4 A")
            newCurve = curveA
        else:
            log.debug("PLP4 B")
            newCurve = curveB
    else:
        curves = [curveA, curveB, curveC]
        minIndex = min((curve.duration, idx) for (idx, curve) in enumerate(curves))[1]
        newCurve = curves[minIndex]
        if minIndex == 0:
            log.debug("PLP4 A")
        elif minIndex == 1:
            log.debug("PLP4 B")
        else:
            log.debug("PLP4 C")
            
    return newCurve
    

def _PLP5(curve, vm, am, delta):
    """(t0 >= delta) and (t1 < delta) and (t2 < delta)
    
    This case is similar to PLP2 and we first follow the procedure of PLP2. However, PLP2A and PLP2B
    can both be infeasible. So we introduct PLP4C where we re-interpolate the trajectory to have
    two-ramp by using PP2 procedure.
    """
    firstRamp = curve[0]
    middleRamp = curve[1]
    lastRamp = curve[2]

    t0 = firstRamp.duration
    t1 = middleRamp.duration
    t2 = lastRamp.duration
    v0 = firstRamp.v0
    v1 = lastRamp.v1
    x0 = firstRamp.x0
    d = curve.d
    d1 = middleRamp.d + lastRamp.d
    vp = middleRamp.v0 # the middle has zero acceleration
    deltaInv = 1.0/delta

    # Try case A.
    a2New = (v1 - vp)*deltaInv
    ramp2A = Ramp(vp, a2New, delta)
    dRem = d - ramp2A.d
    subCurveA = _Compute1DTrajectoryWithDelta(0, dRem, v0, vp, vm, am, delta)
    assert(not subCurveA.IsEmpty()) # this computation should not fail
    if len(subCurveA) == 1:
        curveA = ParabolicCurve([subCurveA[0], ramp2A])
    else:
        # len(subCurveA) == 2 in this case
        curveA = ParabolicCurve([subCurveA[0], subCurveA[1], ramp2A])
    
    # Try case B.
    if FuzzyZero(vp + v1, epsilon):
        curveB = ParabolicCurve()
    else:
        passed = False
        t1New = 2*d1/(vp + v1)
        if (t1New >= delta):
            a1New = (v1 - vp)/t1New
            if abs(a1New) <= am + epsilon:
                ramp1B = Ramp(vp, a1New, t1New)
                curveB = ParabolicCurve([firstRamp, ramp1B])
        if not passed:
            curveB = ParabolicCurve()

    # Try case C.
    # Two-ramp velocity profile where the first ramp saturates the minimum-switch-time constraint
    # and the second ramp saturates the acceleration constraint.
    a0 = lastRamp.a
    k = a0*delta
    vpNew = 0.5*(k - np.sign(a0)*np.sqrt(k**2 + 4*(k*v0 + v1**2) - 8*a0*d))
    if abs(vpNew) > vm + epsilon:
        curveC = ParabolicCurve()
    else:
        t0New = (vpNew - v0)/a0 # a0 is not zero
        if (t0New >= delta):
            # PP2A: no further correction is needed.
            a1New = (v1 - vpNew)*deltaInv
            ramp0C_A = Ramp(v0, a0, t0New, x0)
            ramp1C_A = Ramp(vpNew, a1New, delta)
            curveC_A = ParabolicCurve([ramp0C_A, ramp1C_A])
            curveC = curveC_A
        else:
            # Try case PP1B.
            vpNew = d*deltaInv - 0.5*(v0 + v1)
            a0New = (vpNew - v0)*deltaInv
            a1New = (v1 - vpNew)*deltaInv

            if (abs(a0New) <= am + epsilon) and (abs(a1New) <= am + epsilon) and (abs(vpNew) <= vm + epsilon):
                ramp0C_B = Ramp(v0, a0New, delta, x0)
                ramp1C_B = Ramp(vpNew, a1New, delta)
                curveC_B = ParabolicCurve([ramp0C_B, ramp1C_B])
            else:
                # Case PP2B does not produce any feasible solution
                curveC_B = ParabolicCurve()

            # Try case PP1C.
            if FuzzyZero(v0 + v1, epsilon):
                # Case C does not produce any feasible solution
                curveC_C = ParabolicCurve()
            else:
                tNew = 2*d/(v0 + v1)
                aNew = (v1 - v0)/tNew
                if (abs(aNew) <= am + epsilon) and (tNew >= delta):
                    ramp0C_C = Ramp(v0, aNew, tNew, x0)
                    curveC_C = ParabolicCurve([ramp0C])
                else:
                    # Case C does not produce any feasible solution
                    curveC_C = ParabolicCurve()

            assert (curveC_B.IsEmpty() and curveC_C.IsEmpty())
            if curveC_B.IsEmpty():
                curveC = curveC_C
            elif curveC_C.IsEmpty():
                curveC = curveC_B
            else:
                curveC = curveC_B if curveC_B.duration < curveC_C.duration else curveC_C
                
    # Now compare PLP1A, PLP1B, and PLP1C
    if curveA.IsEmpty() and curveB.IsEmpty() and curveC.IsEmpty():
        assert False
        newCurve = curveA # empty curve
    elif (not curveA.IsEmpty()) and curveB.IsEmpty() and curveC.IsEmpty():
        log.debug("PLP5 A")
        newCurve = curveA
    elif curveA.IsEmpty() and (not curveB.IsEmpty()) and curveC.IsEmpty():
        log.debug("PLP5 B")
        newCurve = curveB
    elif curveA.IsEmpty() and curveB.IsEmpty() and (not curveC.IsEmpty()):
        log.debug("PLP5 C")
        newCurve = curveC
    elif curveA.IsEmpty():
        if curveB.duration <= curveC.duration:
            log.debug("PLP5 B")
            newCurve = curveB
        else:
            log.debug("PLP5 C")
            newCurve = curveC
    elif curveB.IsEmpty():
        if curveA.duration <= curveC.duration:
            log.debug("PLP5 A")
            newCurve = curveA
        else:
            log.debug("PLP5 C")
            newCurve = curveC
    elif curveC.IsEmpty():
        if curveA.duration <= curveB.duration:
            log.debug("PLP5 A")
            newCurve = curveA
        else:
            log.debug("PLP5 B")
            newCurve = curveB
    else:
        curves = [curveA, curveB, curveC]
        minIndex = min((curve.duration, idx) for (idx, curve) in enumerate(curves))[1]
        newCurve = curves[minIndex]
        if minIndex == 0:
            log.debug("PLP5 A")
        elif minIndex == 1:
            log.debug("PLP5 B")
        else:
            log.debug("PLP5 C")
            
    return newCurve


def _PLP6(curve, vm, am, delta):
    """(t0 < delta) and (t1 >= delta) and (t2 < delta)
    
    We explore four possibilities
    
    A: stretch both the first and the last ramps.  
    B: stretch only the first ramp. (The remaining displacement is taken care by
       basically using Compute1DTrajectoryWithDelta.)
    C: stretch only the last ramp, similar to B.
    D: merge everything into one ramp (because sometimes the first and the last ramps
       are both too short).
    """
    firstRamp = curve[0]
    middleRamp = curve[1]
    lastRamp = curve[2]

    t0 = firstRamp.duration
    t1 = middleRamp.duration
    t2 = lastRamp.duration
    v0 = firstRamp.v0
    v1 = lastRamp.v1
    x0 = firstRamp.x0
    d = curve.d
    d0 = firstRamp.d + middleRamp.d
    vp = middleRamp.v0 # the middle has zero acceleration
    deltaInv = 1.0/delta
    
    newFirstRamp = Ramp(v0, (vp - v0)*deltaInv, delta, x0)
    newLastRamp = Ramp(vp, (v1 - vp)*deltaInv, delta)
    
    # Try case A.
    dRemA = d - (newFirstRamp.d + newLastRamp.d)
    t1New = dRemA/vp
    if (t1New > 0):
        newMiddleRamp = Ramp(vp, 0.0, t1New)
        curveA = ParabolicCurve([newFirstRamp, newMiddleRamp, newLastRamp])
        if (t1New < delta):
            curveA = _FixSwitchTimeThreeRamps(curveA, vm, am, delta) # go to PLP3
    else:
        curveA = ParabolicCurve()

    # Try case B.
    # B1
    dRemB = d - newFirstRamp.d
    subCurveB1 = _Compute1DTrajectoryWithDelta(0, dRemB, vp, v1, vm, am, delta)
    if len(subCurveB1) == 1:
        curveB1 = ParabolicCurve([newFirstRamp, subCurveB1[0]])
    else:
        # len(subCurveB) == 2 in this case
        curveB1 = ParabolicCurve([newFirstRamp, subCurveB1[0], subCurveB1[1]])

    # B2
    curveB2 = _PP1(curve, vm, am, delta)
    
    if curveB2.IsEmpty() or abs(curveB2[0].v1) > vm:
        curveB = curveB1
    else:
        if curveB1.duration <= curveB2.duration:
            curveB = curveB1
        else:
            curveB = curveB2

    # Try case C.
    # C1
    dRemC = d - newLastRamp.d
    subCurveC1 = _Compute1DTrajectoryWithDelta(x0, x0 + dRemC, v0, vp, vm, am, delta)
    if len(subCurveC1) == 1:
        curveC1 = ParabolicCurve([subCurveC1[0], newLastRamp])
    else:
        # len(subCurveC1) == 2 in this case
        curveC1 = ParabolicCurve([subCurveC1[0], subCurveC1[1], newLastRamp])

    # C2
    curveC2 = _PP2(curve, vm, am, delta)
    
    if curveC2.IsEmpty() or abs(curveC2[0].v1) > vm:
        curveC = curveC1
    else:
        if curveC1.duration <= curveC2.duration:
            curveC = curveC1
        else:
            curveC = curveC2
        
    # Try case D.
    if FuzzyZero(v0 + v1, epsilon):
        curveD = ParabolicCurve()
    else:
        tNew = 2*d/(v0 + v1)
        if tNew <= 0:
            curveD = ParabolicCurve()
        else:
            aNew = (v1 - v0)/tNew
            if abs(aNew) > am + epsilon:
                curveD = ParabolicCurve()
            else:
                ramp0D = Ramp(v0, aNew, tNew, x0)
                curveD = ParabolicCurve([ramp0D])

    # Now compare every case
    if not curveA.IsEmpty():
        if (curveA.duration <= curveB.duration) and (curveA.duration <= curveC.duration) and (curveA[1].duration >= delta):
            if curveD.IsEmpty():
                # Check before returning
                log.debug("PLP6 A")
                return curveA
            else:
                if curveA.duration <= curveD.duration:
                    # Check before returning
                    log.debug("PLP6 A")
                    return curveA
        
    if (curveB.duration <= curveC.duration):
        if curveD.IsEmpty():
            # Check before returning
            log.debug("PLP6 B")
            return curveB
        else:
            if curveB.duration <= curveD.duration:
                # Check before returning
                log.debug("PLP6 B")
                return curveB
    else:
        if curveD.IsEmpty():
            # Check before returning
            log.debug("PLP6 C")
            return curveC
        else:
            if curveC.duration <= curveD.duration:
                # Check before returning
                log.debug("PLP6 C")
                return curveC

    # Check before returning
    log.debug("PLP6 D")
    return curveD


def _PLP7(curve, vm, am, delta):
    """(t0 < delta) and (t1 < delta) and (t2 < delta)  
    
    """
    firstRamp = curve[0]
    middleRamp = curve[1]
    lastRamp = curve[2]

    t0 = firstRamp.duration
    t1 = middleRamp.duration
    t2 = lastRamp.duration
    v0 = firstRamp.v0
    v1 = lastRamp.v1
    x0 = firstRamp.x0
    d = curve.d
    d0 = firstRamp.d + middleRamp.d
    vp = middleRamp.v0 # the middle has zero acceleration
    deltaInv = 1.0/delta
    
    # A: (delta, delta, delta)
    curveA = Recompute1DTrajectoryThreeRamps(curve, delta, delta, delta, vm, am)

    # B: fix using PP3
    curveB = _PP3(curve, vm, am, delta)

    # C: exceptional case
    if FuzzyZero(v0, epsilon) or FuzzyZero(v1, epsilon):
        if abs(vp - v0) > abs(vp - v1):
            dNew = 0.5*(vp + v0)*delta
            t1New = (2*d - 2*dNew)/(vp + v1)
            if t1New > 0:
                a0New = (vp - v0)*deltaInv
                a1New = (v1 - vp)/t1New
                ramp0C = Ramp(v0, a0New, delta, x0)
                ramp1C = Ramp(vp, a1New, t1New)
                curveC = ParabolicCurve([ramp0C, ramp1C])
            else:
                curveC = ParabolicCurve()
        else:
            dNew = 0.5*(vp + v1)*delta
            t0New = (2*d - 2*dNew)/(vp + v0)
            if t0New > 0:
                a0New = (vp - v0)/t0New
                a1New = (v1 - vp)*deltaInv
                ramp0C = Ramp(v0, a0New, t0New, x0)
                ramp1C = Ramp(vp, a1New, delta)
                curveC = ParabolicCurve([ramp0C, ramp1C])
            else:
                curveC = ParabolicCurve()
    else:
        curveC = ParabolicCurve()
    
    # Now compare all cases
    if curveA.IsEmpty() and curveB.IsEmpty() and curveC.IsEmpty():
        assert False
        newCurve = curveA # empty curve
    elif (not curveA.IsEmpty()) and curveB.IsEmpty() and curveC.IsEmpty():
        log.debug("PLP7 A")
        newCurve = curveA
    elif curveA.IsEmpty() and (not curveB.IsEmpty()) and curveC.IsEmpty():
        log.debug("PLP7 B")
        newCurve = curveB
    elif curveA.IsEmpty() and curveB.IsEmpty() and (not curveC.IsEmpty()):
        log.debug("PLP7 C")
        newCurve = curveC
    elif curveA.IsEmpty():
        if curveB.duration <= curveC.duration:
            log.debug("PLP7 B")
            newCurve = curveB
        else:
            log.debug("PLP7 C")
            newCurve = curveC
    elif curveB.IsEmpty():
        if curveA.duration <= curveC.duration:
            log.debug("PLP7 A")
            newCurve = curveA
        else:
            log.debug("PLP7 C")
            newCurve = curveC
    elif curveC.IsEmpty():
        if curveA.duration <= curveB.duration:
            log.debug("PLP7 A")
            newCurve = curveA
        else:
            log.debug("PLP7 B")
            newCurve = curveB
    else:
        curves = [curveA, curveB, curveC]
        minIndex = min((curve.duration, idx) for (idx, curve) in enumerate(curves))[1]
        newCurve = curves[minIndex]
        if minIndex == 0:
            log.debug("PLP7 A")
        elif minIndex == 1:
            log.debug("PLP7 B")
        else:
            log.debug("PLP7 C")
    # Check before returning
    return newCurve
