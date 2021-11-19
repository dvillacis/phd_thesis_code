import numpy as np
from numpy.linalg import norm, lstsq
from scipy.optimize._lsq.dogbox import dogleg_step,find_intersection
from scipy.optimize._lsq.common import print_iteration_nonlinear, step_size_to_bound,in_bounds,update_tr_radius,evaluate_quadratic,build_quadratic_1d,minimize_quadratic_1d,check_termination,print_header_nonlinear
from scipy.optimize import BFGS

def nsdogbox(fun,grad,x0,lb,ub,initial_radius=1.0,verbose=0,xtol=1e-5,ftol=1e-5,gtol=1e-5,max_nfev=100):

    on_bound = np.zeros_like(x0,dtype=int)
    on_bound[np.equal(x0,lb)] = -1
    on_bound[np.equal(x0,ub)] = 1

    x = x0
    step = np.empty_like(x0)
    radius = initial_radius

    termination_status = None
    iteration = 0
    step_norm = None
    actual_reduction = None
    nfev = 1
    B = BFGS()
    B.initialize(len(x),'hess')
    scale = np.ones_like(x0)
    scaleinv = 1/scale

    if verbose == 2:
        print_header_nonlinear()
    f = fun(x)
    g = grad(x)

    while True:
        active_set = on_bound * g < 0
        free_set = ~active_set

        g_free = g[free_set]
        g_full = g.copy()
        g[active_set] = 0

        g_norm = norm(g,ord=np.inf)
        if g_norm < gtol:
            termination_status = 1

        if verbose == 2:
            print_iteration_nonlinear(iteration,nfev,fun(x),actual_reduction,step_norm,g_norm)

        if termination_status is not None or nfev == max_nfev:
            break

        x_free = x[free_set]
        lb_free = lb[free_set]
        ub_free = ub[free_set]
        scale_free = scale[free_set]
        B_free = B.get_matrix()[:,free_set]

        print(B_free.shape,g_free.shape)
        newton_step = lstsq(B_free,-g_free,rcond=-1)[0]
        a,b = build_quadratic_1d(B_free,g_free,-g_free)
        actual_reduction = -1.0
        while actual_reduction <= 0 and nfev < max_nfev:
            tr_bounds = radius * scale_free
            
            step_free, on_bound_free, tr_hit = dogleg_step(x_free,newton_step,g_free,a,b,tr_bounds,lb_free,ub_free)
            
            step.fill(0.0)
            step[free_set] = step_free

            predicted_reduction = -evaluate_quadratic(B_free,g_free,step_free)

            x_new = np.clip(x + step, lb, ub)

            f_new = fun(x_new)
            nfev += 1

            step_h_norm = norm(step * scaleinv, ord=np.inf)

            if not np.all(np.isfinite(f_new)):
                radius = 0.25 * step_h_norm
                continue

            actual_reduction = f-f_new

            radius, ratio = update_tr_radius(radius,actual_reduction,predicted_reduction,step_h_norm,tr_hit)

            step_norm = norm(step)
            termination_status = check_termination(actual_reduction,f,step_norm,norm(x),ratio,ftol,xtol)

            if termination_status is not None:
                break
        
        if actual_reduction > 0:
            on_bound[free_set] = on_bound_free

            x = x_new

            mask = on_bound == -1
            x[mask] = lb[mask]
            mask = on_bound == 1
            x[mask] = ub[mask]

            f = f_new
            f_true = f.copy()

            B.update(step,g-grad(x_new))

            g = grad(x_new)
        else:
            step_norm = 0
            actual_reduction = 0

        iteration += 1
    if termination_status is None:
        termination_status = 0

    return x