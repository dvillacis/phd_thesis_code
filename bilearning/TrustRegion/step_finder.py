
import numpy as np
import scipy
import scipy.linalg

def cauchy_point_step_finder(gx, b, delta):
    gt_b_g = np.dot(gx,b.dot(gx))
    g_norm = np.linalg.norm(gx)
    if gt_b_g <= 0:
        taw = delta/g_norm
    else:
        taw = min(g_norm**2/gt_b_g, delta/g_norm)
    print(delta/g_norm,gx,g_norm)
    return -taw*gx

def solve_taw_for_dogleg(pu, pb, delta):
    a = np.dot(pu-pb, pu-pb)
    b = -2 * (2*np.dot(pu, pu) + np.dot(pb, pb) - 3*np.dot(pu, pb))
    c = np.dot(2*pu-pb, 2*pu-pb) - delta**2
    d = np.sqrt(b**2 - 4*a*c)
    t1 = (-b + d) / (2*a)
    t2 = (-b - d) / (2*a)
    if 0 <= t1 <= 2:
        if 0 <= t2 <= 2:
            return min(t1, t2)
        return t1
    elif 0 <= t2 <= 2:
        return t2
    else:
        raise ArithmeticError('Taw is not in [0,2]: %d %d', t1, t2)


def dogleg_step_finder(gx, b, delta):
    pb = -scipy.linalg.solve(b.get_matrix(),gx)
    if np.linalg.norm(pb) <= delta:
        return pb
    pu = - (np.dot(gx,gx) / (np.dot(gx,b.dot(gx)))) * gx
    if np.linalg.norm(pu) >= delta:
        return (delta / np.linalg.norm(pu)) * (1-1e-3) * pu
    taw = solve_taw_for_dogleg(pu, pb, delta)
    if taw <= 1:
        return taw * pu
    else:
        return pu + (taw - 1)*(pb - pu)