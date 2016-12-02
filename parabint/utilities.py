import numpy as np
epsilon = 1e-8
inf = 1e300


def FuzzyEquals(a, b, epsilon):
    return abs(a - b) <= epsilon


def FuzzyZero(a, epsilon):
    return abs(a) <= epsilon


def SolveLinearInEq(a, b, epsilon, xmin, xmax):
    """This functions safely solves for x \in [xmin, xmax] such that |ax - b| <= epsilon*max(|a|, |b|).

    Return [result, x]
    """
    if (a < 0):
        return SolveLinearInEq(-a, -b, epsilon, xmin, xmax)
    epsilonScaled = epsilon*max(a, abs(b))

    if (xmin == -inf) and (xmax == inf):
        if (a == 0):
            x = 0.0
            result = abs(b) <= epsilonScaled
            return [result, x]

        x = b/a
        return [True, x]

    axmin = a*xmin
    axmax = a*xmax
    if not (b + epsilonScaled >= axmin and b - epsilonScaled <= axmax):
        # Ranges do not intersect
        return [False, 0.0]

    if not (a == 0):
        x = b/a
        if (xmin <= x) and (x <= xmax):
            return [True, x]

    if abs(0.5*(axmin + axmax) - b) <= epsilonScaled:
        x = 0.5*(xmin + xmax)
        return [True, x]

    if abs(axmax - b) <= epsilonScaled:
        x = xmax
        return [True, x]

    x = xmin
    return [True, x]


def SolveBoundedInEq(a, b, l, u):
    """Solve inequalities of the form
               l <= ax + b <= u.
    
    The function returns [solved, xl, xu] such that ax + b \in [l, u] for all x \in [xl, xu] if solved.
    """
    if l > u:
        return [False, inf, -inf]

    if FuzzyZero(a, epsilon):
        if (b >= l) and (b <= u):
            return [True, l, u]
        else:
            return [False, inf, -inf]

    l -= b
    u -= b
    aInv = 1.0/a
    if a > 0:
        return [True, l*aInv, u*aInv]
    else:
        return [True, u*aInv, l*aInv]


def BrakeTime(x, v, xbound):
    [solved, t] = SolveLinearInEq(v, 2*(xbound - x), epsilon, 0, inf)
    if not solved:
        log.debug("Cannot solve for braking time from the equation: {0}*t - {1} = 0".format(v, 2*(xbound - x)))
        t = 0
    return t


def BrakeAccel(x, v, xbound):
    coeff0 = 2*(xbound - x)
    coeff1 = v*v
    [solved, a] = SolveLinearInEq(coeff0, -coeff1, epsilon, -inf, inf)
    if not solved:
        log.debug("Cannot solve for braking acceleration from the equation: {0}*a + {1} = 0".format(coeff0, coeff1))
        a = 0
    return a
    

def Swap(a, b):
    return [b, a]
