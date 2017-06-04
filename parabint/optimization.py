import numpy as np
import pyipopt
import random
rng = random.SystemRandom()
from .utilities import epsilon

pyipopt.set_loglevel(0)
inf = pyipopt.NLP_UPPER_BOUND_INF

def SolveTwoRampTrajectory(d, v0, v1, vm, am, delta, max_iter=5000, tol=epsilon, print_level=0):
    """Given the boundary conditions for the trajectory, find a fastest
    two-ramp trajectory satisfying velocity, acceleration, and the
    minimum-switch-time constraint.

    Optimization variable: x = [t0, t1, vp]

    Inequality constarints:

    -am <= (vp - v0)/t0 <= am                 ---(1)
    -am <= (v1 - vp)/t1 <= am                 ---(2)

    Equality constraint:

    d = 0.5*(v0 + vp)*t0 + 0.5*(v1 + vp)*t1   ---(3)

    """
    nvar = 3
    ncon = 3
    nnzj = 9
    nnzh = 6

    # Bounds on the variables
    xl = np.array([delta, delta, vm])
    xu = np.array([INF, INF, vm])
    x0 = [2*rng.random() -1 for _ in xrange(nvar)] # randomly select an initial guess

    # Bounds on the constraints
    gl = np.array([-am, -am, d])
    gu = np.array([am, am, d])

    # Objective function f(x)
    def eval_f(x, user_data=None):
        return x[0] + x[1]

    # Gradient of f(x)
    def eval_grad_f(x, user_data=None):
        return np.array([1., 1., 0.])

    # Constraint function g(x)
    def eval_g(x, user_data=None):
        return np.array([(x[2] - v0)/x[0],
                         (v1 - x[2])/x[0],
                         0.5*(v0 + x[2])*x[0] + 0.5*(v1 + x[2])*x[1]])

    # Jacobian of g(x)
    def eval_jacobian_g(x, flag, user_data=None):
        if flag:
            # Return the position of each number in Jacobian
            return (np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                    np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]))
        else:
            return np.array([-(x[2] - v0)/(x[0]**2),
                             0.0,
                             1.0/x[0],
                             0.0,
                             -(v1 - x[2])/(x[1]**2),
                             -1.0/x[1],
                             0.5*(v0 + x[2]),
                             0.5*(v1 + x[2]),
                             0.5*(x[0] + x[1])])

    # Hessian of f(x)
    def eval_h(x, lagrange, obj_factor, flag, user_data=None):
        if flag:
            return (np.array([0, 0, 1, 0, 1, 2]),
                    np.array([0, 1, 1, 2, 2, 2]))
        else:
            values = np.zeros(nnzh)
            
            values[0] += lagrange[0]*(2*(x[2] - v0)/(x[0]**3))
            values[3] += lagrange[0]*(-1/(x[0]**2))
            
            values[2] += lagrange[1]*(2*(v1 - x[2])/(x[1]**3))
            values[4] += lagrange[1]*(1/(x[1]**2))
            
            values[3] += lagrange[2]*0.5
            values[4] += lagrange[2]*0.5

            return values

    # Initialize a non-linear program
    nlp = pyipopt.create(nvar, xl, xu, ncon, gl, gu, nnzj, nnzh, 
                         eval_f, eval_grad_f, eval_g, eval_jacobian_g, eval_h)
    nlp.int_option('max_iter', max_iter)
    nlp.num_option('tol', tol)
    nlp.int_option('print_level', print_level)    

    x, zl, zu, constraint_multipliers, obj, status = nlp.solve(x0)
    nlp.close()
    
    return x, status


def SolveThreeRampTrajectory(d, v0, v1, vm, am, delta, max_iter=5000, tol=epsilon, print_level=0):
    """
    Given the boundary conditions for the trajectory, find a fastest
    two-ramp trajectory satisfying velocity, acceleration, and the
    minimum-switch-time constraint.

    Optimization variable: x = [t0, t1, t2, vp0, vp1]

    Inequality constarints:

    -am <= (vp - v0)/t0   <= am                 ---(1)
    -am <= (vp1 - vp0)/t1 <= am                 ---(2)
    -am <= (v1 - vp1)/t2  <= am                 ---(3)

    Equality constraint:

    d = 0.5*(v0 + vp0)*t0 + 0.5*(vp0 + vp1)*t1 + 0.5*(vp1 + v1)*t2   ---(4)
    """
    nvar = 5
    ncon = 4
    nnzj = 20
    nnzh = 15

    # Bounds on the variables
    xl = np.array([delta, delta, delta, -vm, -vm])
    xu = np.array([INF, INF, INF, vm, vm])
    x0 = [2*rng.random() -1 for _ in xrange(nvar)] # randomly select an initial guess

    # Bounds on the constraints
    gl = np.array([-am, -am, -am, d])
    gu = np.array([am, am, am, d])

    # Objective function f(x)
    def eval_f(x, user_data=None):
        return x[0] + x[1] + x[2]

    # Gradient of f(x)
    def eval_grad_f(x, user_data=None):
        return np.array([1., 1., 1., 0., 0.])

    # Constraint function g(x)
    def eval_g(x, user_data=None):
        return np.array([(x[3] - v0)/x[0],
                         (x[4] - x[3])/x[1],
                         (v1 - x[4])/x[2],
                         0.5*(v0 + x[3])*x[0] + 0.5*(x[3] + x[4])*x[1] + 0.5*(x[4] + v1)*x[2]])

    # Jacobian of g(x)
    def eval_jacobian_g(x, flag, user_data=None):
        if flag:
            return (np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]),
                    np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]))
        else:
            return np.array([-(x[3] - v0)/(x[0]**2),
                             0.0,
                             0.0,
                             1.0/x[0],
                             0.0,
                             0.0,
                             -(x[4] - x[3])/(x[1]**2),
                             0.0,
                             -1.0/x[1],
                             1.0/x[1],
                             0.0,
                             0.0,
                             -(v1 - x[4])/(x[2]**2),
                             0.0,
                             -1.0/x[2],
                             0.5*(v0 + x[3]),
                             0.5*(x[3] + x[4]),
                             0.5*(x[4] + v1),
                             0.5*(x[0] + x[1]),
                             0.5*(x[1] + x[2])])

    # Hessian of f(x)
    def eval_h(x, lagrange, obj_factor, flag, user_data=None):
        if flag:
            return (np.array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4]),
                    np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4]))
        else:
            values = np.zeros(nnzh)

            values[0]  += lagrange[0]*(2.0*(x[3] - v0)/(x[0]**3))
            values[6]  += lagrange[0]*(-1.0/(x[0]**2))
            
            values[2]  += lagrange[1]*(2.0*(x[4] - x[3])/(x[1]**3))
            values[7]  += lagrange[1]*(1.0/(x[1]**2))
            values[11] += lagrange[1]*(-1.0/(x[1]**2))
            
            values[5]  += lagrange[2]*(2.0*(v1 - x[4])/(x[3]**2))
            values[12] += lagrange[2]*(1.0/(x[2]**2))
            
            values[6]  += lagrange[3]*0.5
            values[7]  += lagrange[3]*0.5
            values[11] += lagrange[3]*0.5
            values[12] += lagrange[3]*0.5
                                      
            return values

    # Initialize a non-linear program
    nlp = pyipopt.create(nvar, xl, xu, ncon, gl, gu, nnzj, nnzh, 
                         eval_f, eval_grad_f, eval_g, eval_jacobian_g, eval_h)
    nlp.int_option('max_iter', max_iter)
    nlp.num_option('tol', tol)
    nlp.int_option('print_level', print_level)    

    x, zl, zu, constraint_multipliers, obj, status = nlp.solve(x0)
    nlp.close()
    
    return x, status
