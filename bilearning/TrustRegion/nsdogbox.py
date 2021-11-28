import numpy as np
from numpy.linalg import norm, lstsq
from scipy.optimize._lsq.dogbox import dogleg_step,find_intersection
from scipy.optimize._lsq.common import step_size_to_bound,in_bounds,update_tr_radius,evaluate_quadratic,build_quadratic_1d,minimize_quadratic_1d,check_termination
from scipy.optimize import BFGS, SR1
from scipy.optimize import OptimizeResult

def print_iteration_dogbox(iteration, nfev, cost, cost_reduction,
                              step_norm, optimality,radius):
    if cost_reduction is None:
        cost_reduction = " " * 15
    else:
        cost_reduction = "{0:^15.2e}".format(cost_reduction)

    if step_norm is None:
        step_norm = " " * 15
    else:
        step_norm = "{0:^15.2e}".format(step_norm)

    print("{0:^15}{1:^15}{2:^15.4e}{3}{4}{5:^15.2e}{6:^15.2e}"
          .format(iteration, nfev, cost, cost_reduction,
                  step_norm, optimality,radius))


def print_header_dogbox():
    print("{0:^15}{1:^15}{2:^15}{3:^15}{4:^15}{5:^15}{6:^15}"
          .format("Iteration", "Total nfev", "Cost", "Cost reduction",
                  "Step norm", "Optimality", "TR-Radius"))

def nsdogbox(fun,grad,reg_grad,x0,lb=None,ub=None,initial_radius=None,threshold_radius=1e-5,verbose=0,xtol=1e-8,ftol=1e-8,gtol=1e-5,max_nfev=10000):

    if not lb:
        lb = 1e-12*np.ones_like(x0)
    if not ub:
        ub = np.ones_like(x0) * np.inf

    on_bound = np.zeros_like(x0,dtype=int)
    on_bound[np.equal(x0,lb)] = -1
    on_bound[np.equal(x0,ub)] = 1

    x = x0
    step = np.empty_like(x0)
    radius = initial_radius
    if radius == None:
        radius = norm(x0, ord=np.inf)

    termination_status = None
    iteration = 0
    step_norm = None
    actual_reduction = None
    nfev = 1
    njev = 0
    n_reg_jev = 0
    B = BFGS(init_scale=0.01)
    # B = BFGS()
    B.initialize(len(x),'hess')
    scale = np.ones_like(x0)
    scaleinv = 1/scale

    if verbose == 2:
        print_header_dogbox()
    f = fun(x)
    if radius >= threshold_radius:
        njev += 1
        g = grad(x)
    else:
        n_reg_jev += 1
        g = reg_grad(x)

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
            print_iteration_dogbox(iteration,nfev,fun(x),actual_reduction,step_norm,g_norm,radius)

        if termination_status is not None or nfev == max_nfev:
            break

        x_free = x[free_set]
        lb_free = lb[free_set]
        ub_free = ub[free_set]
        scale_free = scale[free_set]
        B_free = B.get_matrix()[:,free_set]

        # print(B_free.shape,g_free.shape)
        newton_step = lstsq(B_free,-g,rcond=-1)[0]
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

            # if radius < 1e-6:
            #     B.initialize(len(x),'hess')
                #radius = norm(x)

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

            njev += 1
            if radius >= threshold_radius:
                B.update(step,g-grad(x_new))
                njev +=1
                g = grad(x_new)
            else:
                B.update(step,g-reg_grad(x_new))
                n_reg_jev += 1
                g = reg_grad(x_new)
        else:
            B.initialize(len(x),'hess')
            step_norm = 0
            actual_reduction = 0

        iteration += 1
    if termination_status is None:
        termination_status = 0

    return OptimizeResult(
        x=x, fun=f, jac=g, grad=g_full, optimality=g_norm,
        active_mask=on_bound, nfev=nfev, njev=njev, n_reg_jev=n_reg_jev, status=termination_status, message=termination_status)