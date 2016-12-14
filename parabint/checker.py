from .utilities import epsilon, inf, FuzzyEquals, FuzzyZero
import logging
logging.basicConfig(format='[%(levelname)s] [%(name)s: %(funcName)s] %(message)s', level=logging.DEBUG)
log = logging.getLogger(__name__)

# PCR: ParabolicCheckReturn
PCR_Normal = 0
PCR_NegativeDuration = 1
PCR_XBoundViolated = 2
PCR_VBoundViolated = 3
PCR_ABoundViolated = 4
PCR_XDiscrepancy = 5
PCR_VDiscrepancy = 6
PCR_DurationDiscrepancy = 7


def _GetPeaks(x0, x1, v0, v1, a, t):
    """Calculate the maximum and minimum displacement occuring betweeb time 0 and t given (x0, x1, v0,
    v1, a).

    """
    if FuzzyZero(a, epsilon):
        if v0 > 0:
            bmin = x0
            bmax = x1
        else:
            bmin = x1
            bmax = x0
        return bmin, bmax

    if x0 > x1:
        curMin = x1
        curMax = x0
    else:
        curMin = x0
        curMax = x1

    tDeflection = -v0/a # the time when velocity crosses zero
    if (tDeflection <= 0) or (tDeflection >= t):
        bmin = curMin
        bmax = curMax
        return bmin, bmax

    xDeflection = x0 + 0.5*v0*tDeflection
    bmin = min(curMin, xDeflection)
    bmax = max(curMax, xDeflection)
    return bmin, bmax
    

def CheckSegment(x0, x1, v0, v1, a, t, xmin, xmax, vm, am):
    """
    """
    if t < -epsilon:
        log.warn("PCR_NegativeDuration: duration = {0}".format(t))
        return PCR_NegativeDuration
    if not FuzzyEquals(v1, v0 + a*t, epsilon):
        v1_ = v0 + a*t
        log.warn("PCR_VDiscrepancy: v1 = {0}; computed v1 = {1}; diff = {2}".format(v1, v1_, (v1 - v1_)))
        log.warn("Info: x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; a = {4}; duration = {5}; xmin = {6}; xmax = {7}; vm = {8}; am = {9}".format(x0, x1, v0, v1, a, t, xmin, xmax, vm, am))
        return PCR_VDiscrepancy
    if not FuzzyEquals(x1, x0 + t*(v0 + 0.5*a*t), epsilon):
        x1_ = x0 + t*(v0 + 0.5*a*t)
        log.warn("PCR_XDiscrepancy: x1 = {0}; computed x1 = {1}; diff = {2}".format(x1, x1_, (x1 - x1_)))
        log.warn("Info: x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; a = {4}; duration = {5}; xmin = {6}; xmax = {7}; vm = {8}; am = {9}".format(x0, x1, v0, v1, a, t, xmin, xmax, vm, am))
        return PCR_XDiscrepancy
    if xmin == inf and xmax == inf:
        return PCR_Normal
    bmin, bmax = _GetPeaks(x0, x1, v0, v1, a, t)
    if (bmin < xmin - epsilon) or (bmax > xmax + epsilon):
        log.warn("PCR_XBoundViolated: xmin = {0}; bmin = {1}; diff@min = {2}; xmax = {3}; bmax = {4}; diff@max = {5}".format(xmin, bmin, (xmin - bmin), xmax, bmax, (bmax - xmax)))
        log.warn("Info: x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; a = {4}; duration = {5}; xmin = {6}; xmax = {7}; vm = {8}; am = {9}".format(x0, x1, v0, v1, a, t, xmin, xmax, vm, am))
        return PCR_XBoundViolated
    if (abs(v0) > vm + epsilon) or (abs(v1) > vm + epsilon):
        log.warn("PCR_VBoundViolated: vm = {0}; v0 = {1}; v1 = {2}; diff@v0 = {3}; diff@v1 = {4}".format(vm, v0, v1, (abs(v0) - vm), (abs(v1) - vm)))
        log.warn("Info: x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; a = {4}; duration = {5}; xmin = {6}; xmax = {7}; vm = {8}; am = {9}".format(x0, x1, v0, v1, a, t, xmin, xmax, vm, am))
        return PCR_VBoundViolated
    if abs(a) > am + epsilon:
        log.warn("PCR_ABoundViolated: am = {0}; a = {1}; diff = {2}".format(am, a, (abs(a) - am)))
        log.warn("Info: x0 = {0}; x1 = {1}; v0 = {2}; v1 = {3}; a = {4}; duration = {5}; xmin = {6}; xmax = {7}; vm = {8}; am = {9}".format(x0, x1, v0, v1, a, t, xmin, xmax, vm, am))
        return PCR_ABoundViolated
    return PCR_Normal


def CheckRamp(ramp, xmin, xmax, vm, am):
    """
    """
    return CheckSegment(ramp.x0, ramp.x1, ramp.v0, ramp.v1, ramp.a, ramp.duration, xmin, xmax, vm, am)


def CheckParabolicCurve(curve, xmin, xmax, vm, am, x0, x1, v0, v1):
    """
    """
    # Check the first ramp
    if not FuzzyEquals(curve[0].x0, x0, epsilon):
        log.warn("PCR_XDiscrepancy: curve[0].x0 = {0}; x0 = {1}; diff = {2}".format(curve[0].x0, x0, curve[0].x0 - x0))
        return PCR_XDiscrepancy
    if not FuzzyEquals(curve[0].v0, v0, epsilon):
        log.warn("PCR_VDiscrepancy: curve[0].v0 = {0}; v0 = {1}; diff = {2}".format(curve[0].v0, v0, curve[0].v0 - v0))
        return PCR_VDiscrepancy
    ret = CheckRamp(curve[0], xmin, xmax, vm, am)
    if not (ret == PCR_Normal):
        log.warn("curve[0] does not pass CheckRamp")
        return ret
    
    for iramp in xrange(1, len(curve) - 1):
        if not FuzzyEquals(curve[iramp - 1].x1, curve[iramp].x0, epsilon):
            log.warn("PCR_XDiscrepancy: curve[{0}].x1 != curve[{1}].x0; {2} != {3}; diff = {4}".format(iramp - 1, iramp, curve[iramp - 1].x1, curve[iramp].x0, curve[iramp - 1].x1 - curve[iramp].x0))
            return PCR_XDiscrepancy
        if not FuzzyEquals(curve[iramp - 1].v1, curve[iramp].v0, epsilon):
            log.warn("PCR_VDiscrepancy: curve[{0}].v1 != curve[{1}].v0; {2} != {3}; diff = {4}".format(iramp - 1, iramp, curve[iramp - 1].v1, curve[iramp].v0, curve[iramp - 1].v1 - curve[iramp].v0))
            return PCR_VDiscrepancy
        ret = CheckRamp(curve[iramp], xmin, xmax, vm, am)
        if not (ret == PCR_Normal):
            log.warn("curve[{0}] does not pass CheckRamp".format(iramp))
            return ret

    # Check the last ramp
    if not FuzzyEquals(curve[-1].x1, x1, epsilon):
        log.warn("PCR_XDiscrepancy: curve[{0}].x1 = {1}; x1 = {2}; diff = {3}".format(len(curve) - 1, curve[-1].x0, x0, curve[-1].x1 - x1))
        return PCR_XDiscrepancy
    if not FuzzyEquals(curve[-1].v1, v1, epsilon):
        log.warn("PCR_VDiscrepancy: curve[{0}].v1 = {1}; v1 = {2}; diff = {3}".format(len(curve) - 1, curve[-1].v0, v0, curve[-1].v1 - v1))
        return PCR_VDiscrepancy
    return PCR_Normal
            

def CheckParabolicCurvesND(curvesnd, xminVect, xmaxVect, vmVect, amVect, x0Vect, x1Vect, v0Vect, v1Vect):
    """
    """
    duration = curvesnd.duration
    for (icurve, (curve, xmin, xmax, vm, am, x0, x1, v0, v1)) in enumerate(zip(curvesnd, xminVect, xmaxVect, vmVect, amVect, x0Vect, x1Vect, v0Vect, v1Vect)):
        if not FuzzyEquals(curve.duration, duration, epsilon):
            log.warn("PCR_DurationDiscrepancy: curvesnd.duration = {0}; curvesnd[{1}].duration = {2}; diff = {3}".format(duration, icurve, curve.duration, duration - curve.duration))
            return PCR_DurationDiscrepancy
        ret = CheckParabolicCurve(curve, xmin, xmax, vm, am, x0, x1, v0, v1)
        if not (ret == PCR_Normal):
            log.warn("curvsend[{0}] does not psas CheckParabolicCurve".format(icurve))
            return ret
    return PCR_Normal
