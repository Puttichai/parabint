from parabint import interpolator
from parabint.trajectory import Ramp, ParabolicCurve, ParabolicCurvesND
from matplotlib import pyplot as plt
import random
rng = random.SystemRandom()

_xm = 2.96705973
_vm = 3.92699082
_am = 2.0
delta = 0.2

availableChoices = dict()
availableChoices[len(availableChoices.keys())] = "PP1  : t0 <  delta; t1 >= delta"
availableChoices[len(availableChoices.keys())] = "PP2  : t0 >= delta; t1 <  delta"
availableChoices[len(availableChoices.keys())] = "PP3  : t0 <  delta; t1 <  delta"
availableChoices[len(availableChoices.keys())] = "stop"
msg = "Choose a test case"
for (key, val) in availableChoices.iteritems():
    msg += "\n{0} : {1}".format(key, val)
msg += "\n\n>> "

def SampleConditions(testcase, delta):
    """This function sample boundary conditions (x0, x1, v0, v1) as well
    as the velocity and acceleration bounds vm and am such that the
    time-optimal parabolic trajectory violates the given
    minimum-switch-time constraint in the given case.

    """
    if testcase == 0:
        t0 = delta + 1
        t1 = 0
    elif testcase == 1:
        t0 = 0
        t1 = delta + 1
    else:
        t0 = delta + 1
        t1 = delta + 1

    # Sample a test case
    passed = False
    while not passed:
        x0 = _xm * rng.uniform(-1, 1)
        x1 = _xm * rng.uniform(-1, 1)
        vm = _vm * rng.uniform(0, 1)
        am = _am * rng.uniform(0, 1)
        v0 = vm * rng.uniform(-1, 1)
        v1 = vm * rng.uniform(-1, 1)

        curve1 = interpolator.Compute1DTrajectory(x0, x1, v0, v1, vm, am)
        if not (len(curve1) == 2):
            continue
        t0 = curve1[0].duration
        t1 = curve1[1].duration
        if testcase == 0:
            passed = t0 < delta and t1 >= delta
        elif testcase == 1:
            passed = t0 >= delta and t1 < delta
        else:
            passed = t0 < delta and t1 < delta
    
    return x0, x1, v0, v1, vm, am

while True:
    try:
        index = int(raw_input(msg))
        if index in availableChoices.keys():
            if index == len(availableChoices.keys()) - 1:
                break
            x0, x1, v0, v1, vm, am = SampleConditions(index, delta)

            curve1 = interpolator.Compute1DTrajectory(x0, x1, v0, v1, vm, am)
            curve2 = interpolator.Compute1DTrajectory(x0, x1, v0, v1, vm, am, delta)
            
            # Visualization
            plt.clf()
            curve1.PlotVel(fignum=1, color='r')
            curve2.PlotVel(fignum=1, color='g')
            
            report = \
            "x0 = {0};\nx1 = {1};\nv0 = {2};\nv1 = {3};\nvm = {4};\nam = {5};\ndelta = {6};\n".\
            format(x0, x1, v0, v1, vm, am, delta)
            print report

    except ValueError:
        print "Please enter a valie choice"


