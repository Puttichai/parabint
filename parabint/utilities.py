from parabint import *
import random
import time
try:
    import openravepy as orpy
    HAS_OPENRAVE = True
except:
    HAS_OPENRAVE = False

MAX_REPEAT_SAMPLING = 100
SHORTCUT_THRESHOLD = 0.001#EPSILON

## OpenRAVE dependent.
def ConvertOpenRAVETrajToRampsListND(openravetraj, vmvect, amvect, delta=0):
    configspec = openravetraj.GetConfigurationSpecification()
    configdof = configspec.GetDOF()

    q_group = configspec.GetGroupFromName('joint_values')
    q_dof = q_group.dof
    q_offset = q_group.offset

    if q_group.interpolation == 'linear':
        hasvelocities = False
    else:
        hasvelocities = True
        try:
            qd_group = configspec.GetGroupFromName('joint_velocities')
        except:
            hasvelocities = False

    if hasvelocities:
        assert(qd_group.interpolation == 'linear')
        qd_dof = qd_group.dof
        qd_offset = qd_group.offset
        dt_offset = configspec.GetGroupFromName('deltatime').offset

    nwaypoints = openravetraj.GetNumWaypoints()
    config_wp = openravetraj.GetWaypoints(0, nwaypoints)
    configlist = [config_wp[configdof*i: configdof*(i + 1)] for i in xrange(nwaypoints)]
    q_wp = [w[q_offset:q_offset + q_dof] for w in configlist]
    if hasvelocities:
        qd_wp = [w[qd_offset:qd_offset + qd_dof] for w in configlist]
        dt_wp = [w[dt_offset] for w in configlist]
        rampslist = ConvertWaypointsWithVelocitiesToRampsListND(q_wp, qd_wp, dt_wp, 
                                                                vmvect, amvect, delta)
    else:
        rampslist = ConvertWaypointsToRampsListND(q_wp, vmvect, amvect, delta)
    
    return rampslist


##
def ConvertWaypointsToRampsListND(waypointslist, vmvect, amvect, delta=0):
    ## merge collinear waypoints
    W = MergeWaypoints(waypointslist)
    nwaypoints = len(W)
    newrampslistnd = RampsListND()
    
    for i in range(nwaypoints - 1):
        q0 = W[i]
        q1 = W[i + 1]
        rampslistnd = InterpolateZeroVelND(q0, q1, vmvect, amvect, delta)
        newrampslistnd.Append(rampslistnd)
    return newrampslistnd


##
def MergeWaypoints(waypointslist):
    nwaypoints = len(waypointslist)
    newwaypointslist = []
    newwaypointslist.append(waypointslist[0])
    
    for i in range(1, nwaypoints):
        if len(newwaypointslist) >= 2:
            qprev1 = newwaypointslist[-1]
            qprev2 = newwaypointslist[-2]
            qcur = waypointslist[i]
            
            dq1 = qcur - qprev1
            dq2 = qcur - qprev2
            
            len_dq1sq = np.dot(dq1, dq1)
            len_dq2sq = np.dot(dq2, dq2)
            
            dotproduct = np.dot(dq1, dq2)
            
            if (abs(dotproduct**2 - len_dq1sq*len_dq2sq) < EPSILON):
                ## new waypoint is collinear with the previous one
                newwaypointslist.pop()
            
            newwaypointslist.append(qcur)                
        else:
            if (np.linalg.norm(waypointslist[i] - newwaypointslist[0]) > EPSILON):
                newwaypointslist.append(waypointslist[i])            
    return newwaypointslist


##
def ConvertWaypointsWithVelocitiesToRampsListND(q_wp, qd_wp, dt_wp,
                                                vmvect, amvect, delta=0):
    nwaypoints = len(q_wp)
    newrampslistnd = RampsListND()

    for i in xrange(nwaypoints - 1):
        q0 = q_wp[i]
        qd0 = qd_wp[i]
        qd1 = qd_wp[i + 1]
        dt = dt_wp[i + 1]
        chunk = []
        for dof in xrange(len(q0)):
            a = (qd1[dof] - qd0[dof])/dt
            ramp = Ramp(qd0[dof], a, dt, x0=q0[dof])
            chunk.append(RampsList([ramp]))
        newrampslistnd.Append(RampsListND.FromChunksList([chunk]))

    return newrampslistnd


## ReplaceRampsListNDSegment replaces a segment from t = t0 to t = t1
## of originalrampslistnd with newsegment
def ReplaceRampsListNDSegment(originalrampslistnd, newsegment, t0, t1):
    assert(originalrampslistnd.ndof == newsegment.ndof)
    assert(t1 > t0)
    
    ndof = originalrampslistnd.ndof
    rampslistnd = []
    for j in range(ndof):
        ## replace each dof one by one
        newrampslist = RampsList()
        rampslist = originalrampslistnd[j]  
        i0, rem0 = rampslist.FindRampIndex(t0)
        i1, rem1 = rampslist.FindRampIndex(t1)
        
        ## check if t0 falls in the first ramp. 
        ## if not, insert ramp 0 to ramp i0 - 1 into newrampslist
        if i0 > 0:
            newrampslist.Append(RampsList(rampslist[0: i0]))
                
        ## remainder ramp 0
        if (abs(rem0) >= EPSILON):
            remramp0 = Ramp(rampslist[i0].v, rampslist[i0].a, rem0)
            ## initial condition has to be set because sometimes remramp0 is
            ## the beginning of the shortcut traj
            remramp0.x0 = rampslist[i0].x0
            newrampslist.Append(RampsList([remramp0]))            
        
        ## insert newsegment
        newrampslist.Append(newsegment[j])
        
        ## remainder ramp 1
        if (abs(rampslist[i1].T - rem1) >= EPSILON):
            remramp1 = Ramp(rampslist[i1].Evald(rem1), rampslist[i1].a,
                            rampslist[i1].T - rem1)
            newrampslist.Append(RampsList([remramp1]))

        ## insert remaining ramps
        if i1 < len(rampslist) - 1:
            newrampslist.Append(RampsList(rampslist[i1 + 1: len(rampslist)]))
            
        rampslistnd.append(newrampslist)

    return RampsListND(rampslistnd)


##
def Shortcut(rampslistnd, vmvect, amvect, delta, shortcutiter, PRINT=True, robot=None,
             ret_data=False):
    rng = random.SystemRandom()
    ndof = rampslistnd.ndof
    ## data collecting
    nsuccessful = 0
    ncollision = 0
    njointlimit = 0
    nnotshorter = 0
    nnotsync = 0
    for it in range(shortcutiter):
        if PRINT:
            print "iteration {0} :".format(it + 1),
        dur = rampslistnd.duration
        
        ## sample two random time instants
        T = 0
        minallowedduration = max(0.04, 5*delta)
        t0 = rng.uniform(0, dur)
        t1 = rng.uniform(0, dur)
        if t1 < t0:
            temp = t1
            t1 = t0
            t0 = temp
        T = t1 - t0
        if (T < minallowedduration):
            if (t0 + minallowedduration < dur):
                t1 = t0 + minallowedduration
            elif (t1 - minallowedduration > 0.0):
                t0 = t1 - minallowedduration
            else:
                t0 = 0.0
                t1 = dur
                
        ## constrain t0 and t1 not to violate the minimum switching
        ## time constraint
        i0 = bisect.bisect_left(rampslistnd.switchpointslist, t0) - 1
        i1 = bisect.bisect_left(rampslistnd.switchpointslist, t1) - 1
        ## snap t0 and t1 to the nearest switching points (if neccessary)
        ## snapping t0
        if (t0 - rampslistnd.switchpointslist[i0] < max(0.008, delta)):
            t0 = rampslistnd.switchpointslist[i0]
        ## snapping t1
        if (rampslistnd.switchpointslist[i1 + 1] - t1 < max(0.008, delta)):
            t1 = rampslistnd.switchpointslist[i1 + 1]

        T = t1 - t0

        q0 = rampslistnd.Eval(t0)
        qd0 = rampslistnd.Evald(t0)
        q1 = rampslistnd.Eval(t1)
        qd1 = rampslistnd.Evald(t1)
        newrampslistnd = InterpolateArbitraryVelND(q0, q1, qd0, qd1, 
                                                   vmvect, amvect, delta)
        if not (newrampslistnd.isvoid):
            ## check the new duration
            Tnew = newrampslistnd.duration
            if (T - Tnew > SHORTCUT_THRESHOLD):
                ## check joint limits
                injointlimits = CheckJointLimits(robot, newrampslistnd)
                if (injointlimits):
                    ## check collision
                    incollision = InCollision(robot, newrampslistnd)
                    if (not incollision):
                        rampslistnd = ReplaceRampsListNDSegment\
                        (rampslistnd, newrampslistnd, t0, t1)
                        
                        nsuccessful += 1
                        if PRINT:
                            print "Successful Shortcut"
                    else:
                        ncollision += 1
                        if PRINT:
                            print "In Collision"
                else:
                    njointlimit += 1
                    if PRINT:
                        print "Not in Joint Limits"
            else:
                nnotshorter += 1
                if PRINT:
                    print "Not Shorter"
        else:
            nnotsync += 1
            if PRINT:
                print "Not Synchronizable"
    
    if ret_data:
        data = [nsuccessful, ncollision, njointlimit, nnotshorter, nnotsync]
        return [rampslistnd, data]
    
    return rampslistnd


## CheckJointLimits
def CheckJointLimits(robot, rampslistnd):
    if not HAS_OPENRAVE:
        return True

    ## user defined function
    injointlimits = False
    jointlimits = robot.GetDOFLimits()[1] ## get joint upper limits
    ndof = rampslistnd.ndof
    for i in range(ndof):
        rampslist = rampslistnd[i]
        nramps = len(rampslist)
        for j in range(nramps):
            ramp = rampslist[j]
            if (abs(ramp.a) < EPSILON):
                x_extremal = ramp.Eval(ramp.T)
            else:
                t_extremal = -ramp.v/ramp.a
                if (t_extremal < 0) or (t_extremal > ramp.T):
                    x_extremal = ramp.Eval(ramp.T)
                else:
                    x_extremal = ramp.Eval(t_extremal)
            if (abs(x_extremal) > jointlimits[i]):
                return injointlimits
    injointlimits = True
    return injointlimits


## CheckCollision checks collision for the robot along rampslistnd
def InCollision(robot, rampslistnd, checkcollisionstep=0.005):
    if not HAS_OPENRAVE:
        return True

    ## user defined function
    env = robot.GetEnv()
    t = 0
    incollision = False
    while t < rampslistnd.duration:
        robot.SetActiveDOFValues(rampslistnd.Eval(t))
        incollision = env.CheckCollision(robot)
        if incollision:
            return incollision
        t += checkcollisionstep
    robot.SetActiveDOFValues(rampslistnd.Eval(rampslistnd.duration))
    incollision = env.CheckCollision(robot)
    
    return incollision
