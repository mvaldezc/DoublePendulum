import torch

# define the running cost function
def running_cost_func(curr_x, curr_u, xstar, Q, R):
    running_cost = 0.5*(curr_x - xstar).T @ Q @ (curr_x - xstar) + 0.5 * curr_u * R * curr_u
    return running_cost

# define the terminal cost function
def term_cost_func(final_x, xstar, Qf):
    term_cost = 0.5 * (final_x - xstar).T @ Qf @ (final_x - xstar)
    return term_cost

# define the total cost function
# TODO: sometimes running cost only goes from k to the final value
def total_cost_func(xs, us, xstar, Q, R, Qf):
    N = xs.shape[0] # should be 51
    total_cost = 0.0

    for i in range(N-1):
        curr_x = xs[i, :]
        curr_u = us[i, :]
        total_cost += running_cost_func(curr_x, curr_u, xstar, Q, R)

    final_x = xs[-1, :]
    total_cost += term_cost_func(final_x, xstar, Qf)
    return total_cost

