import torch
from pendulum import *

    # perform rollouts to obtain the nominal state and control trajectories
def obtain_nominal_trajectory(current_state, N, Tf):
    xs_nom, us_nom = rollout_dynamics(N, current_state)

    # plot the nominal trajectory over time
    # t_vec = np.linspace(0, Tf, N)
    # fig, axs = plt.subplots(3,1,sharex=True)
    # lin0, = axs[0].plot(t_vec, xs_nom[:,0], lw=2) 
    # lin1, = axs[1].plot(t_vec, xs_nom[:,1], lw=2)
    # lin2, = axs[2].plot(t_vec, xs_nom[:,2], lw=2)
    # axs[0].set_title('Nominal trajectories over time')
    # axs[0].set_ylabel("X position")
    # axs[1].set_ylabel("Theta 1 (rad)")
    # axs[2].set_ylabel("Theta 2 (rad)")
    # axs[-1].set_xlabel("time steps")  

    # # plot the nominal control trajectory
    # plt.figure()
    # t_vec = np.linspace(0, Tf, N-1)
    # lin0, = plt.plot(t_vec, us_nom[:,0], lw=1)
    # plt.title('Nominal control inputs over time')
    # plt.ylabel("control inputs (N)")
    # plt.xlabel("time (s)")

    return xs_nom, us_nom

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

# compute the first derivatives of q
def compute_qx_qu(lx, A, Vx, lu, B):
    qx = lx + A.T @ Vx
    qu = lu + B.T @ Vx
    return qx, qu

# compute the second derivative of q
def compute_qxx_quu(lxx, A, Vxx, luu, B, lux, mu):
    qxx = lxx + A.T @ Vxx @ A 
    quu = luu + B.T @ (Vxx + mu*torch.eye(6)) @ B
    qux = lux + B.T @ (Vxx + mu*torch.eye(6)) @ A
    return qxx, quu, qux

# set up the backward pass function (recursively apply the belman equation)
# proceed backwards in time through the nominal trajectory
def backward_pass(xs_nom, us_nom, xstar, Q, R, mu):

    # get the number of timesteos
    N = xs_nom.shape[0]

    # create a tensor for storing all Vx (jacobian of value function) and Vxx (hessian of the value function) values
    Vxs = torch.zeros((N, 6))
    Vxxs = torch.zeros((N, 6, 6))

    # create a tensor for storing all the gains k and K for the entire trajectory
    ks = torch.ones(N-1, 1, 1) # TODO: initialize to zero?
    Ks = torch.zeros(N-1, 1, 6) # TODO: check the dimensions of these

    # for each value in the nominal trajectory
    for t in reversed(range(1,N,1)): # t goes from N-1 to 1, t-1 goes from N-2 to 0

        # get the current state and control from the nominal trajectory (t-1)
        curr_x = xs_nom[t-1, :].reshape(6,1)
        curr_u = us_nom[t-1, :].reshape(1,1)

        # linearize about the current point in the nominal trajectory (t-1)
        A, B = linearize_dynamics(curr_x.reshape(6,), curr_u.reshape(1,))  
        # print(f'{A=}, {B=}')

        # compute the first and second derivative of the lienarized dynamics (lx, lu, etc) (t-1)
        # print(f'{curr_x.shape=}')
        # print(f'{xstar=}')
        # print(f'{Q=}')
        lx = ((curr_x-xstar.reshape(6,1)).T @ Q.T).reshape(6,1)
        lu = ((curr_u).T @ R.T).reshape(1,1)
        lxx = Q
        luu = R
        lux = 0

        # obtain the first and second derivatives of the value function (Vx') for t-1
        Vx = Vxs[t, :].reshape(6,1)
        Vxx = Vxxs[t, :, :].squeeze(0) # (6,6)

        # compute the first derivatives qx and qu (t-1)
        qx, qu = compute_qx_qu(lx, A, Vx, lu, B)

        # compute the second derivatives Qxx, Quu, Qux (t-1)
        qxx, quu, qux = compute_qxx_quu(lxx, A, Vxx, luu, B, lux, mu)

        # compute the gains k and K (t-1)
        k = -torch.linalg.inv(quu) @ qu # (1, 1)
        K = -torch.linalg.inv(quu) @ qux # (1, 6)

        # store the new gains in the tensor of all the gains
        ks[t-1, :, :] = k
        Ks[t-1, :, :] = K

        # use the new k and K to compute new values for Vx and Vxx 
        Vx = qx + K.T @ quu @ k + K.T @ qu + qux.T @ k # (6, 1) # TODO: not sure if terms with gains should be pos or neg
        Vxx = qxx + K.T @ quu @ K + K.T @ qux + qux.T @ K # (6,6)

        # update the stored Vxs and Vxxs with the new values for Vx and Vxx
        Vxs[t-1, :] = Vx.reshape(1,6)
        Vxxs[t-1, :] = Vxx

    return ks, Ks, quu

    # set up the forward pass function to obtain the new xs and us
def forward_pass(ks, Ks, N, xs_nom, us_nom):
    # save a tensor of all the new states and control inputs
    us = torch.zeros((N-1, 1))
    xs = torch.zeros((N, 6))

    # set the first stae of the new xs equal to the first state of the nominal x
    xs[0, :] = xs_nom[0, :]

    # iterate through all the timepoints
    for t in range(N-1): # t goes from 0 to N-2
        # print(t)
        # get the nominal u, nominal x, and current x for the given timstep
        curr_nom_u = us_nom[t, :].reshape(1,1)
        curr_nom_x = xs_nom[t, :].reshape(6,1)
        curr_x = xs[t, :].reshape(6,1)

        # get the current gains for the given timestep
        k = ks[t, :].reshape(1,1)
        # print(k)
        K = Ks[t, :].reshape(6,1)
        # print(K)

        # calculate delta x from the current x and the nominal x
        delta_x = curr_x - curr_nom_x 
        # calculate delta u using the gains and delta x
        delta_u = k + torch.dot(K.squeeze(-1), delta_x.squeeze(-1))
        # calculate the new u from delta u
        curr_u = curr_nom_u + delta_u # (1,1)
        # print(delta_u)
        # print(curr_u)
        # obtain the next state from the dynamics function
        next_x = dynamics_rk4(curr_x.reshape(1,6), curr_u) # (1,6)

        # save the current u and the next x to the stored vectors of us and xs
        us[t, :] = curr_u
        xs[t+1, :] = next_x

    return us, xs

## perform iLQR
def run_ilqr(current_state, N, Tf, num_iterations, xstar, mu, mu_delta_0, mu_delta, mu_min, Q, R):

    # perform the rollout to obtain the nominal trajectory
    xs_nom, us_nom = obtain_nominal_trajectory(current_state, N, Tf)

    # run loop until convergence is reached
    i = 0
    xs = xs_nom
    us = us_nom

    # matrix to test positive definiteness
    is_pos_def = False

    while i < num_iterations:

        while not is_pos_def:
            # perform the backwards pass
            ks, Ks, quu = backward_pass(xs, us, xstar, Q, R, mu)
            
            if quu <= 0: # not positive definite, increase mu
                is_pos_def = False
                mu_delta = max(mu_delta_0, mu_delta*mu_delta_0)
                mu = max(mu_min, mu*mu_delta)
            else:   # is positive definite, decrease mu
                is_pos_def = True
                mu_delta = min(1/mu_delta_0, mu_delta/mu_delta_0)
                if mu*mu_delta > mu_min:
                    mu = mu*mu_delta
                else:
                    mu = 0

        # perform the forward pass
        us, xs = forward_pass(ks, Ks, N, xs, us)

        i += 1

    # plot the optimal state trajectory over time
    # t_vec_x = np.linspace(0, Tf, N)
    # fig, axs = plt.subplots(3,1,sharex=True)
    # lin0, = axs[0].plot(t_vec_x, xs[:,0], lw=2) 
    # lin1, = axs[1].plot(t_vec_x, xs[:,1], lw=2)
    # lin2, = axs[2].plot(t_vec_x, xs[:,2], lw=2)
    # axs[0].set_title('Optimal state trajectories over time')
    # axs[0].set_ylabel("X position")
    # axs[1].set_ylabel("Theta 1 (rad)")
    # axs[2].set_ylabel("Theta 2 (rad)")
    # axs[-1].set_xlabel("time steps")  

    # # plot the optimal control trajectory
    # plt.figure()
    # t_vec_u = np.linspace(0,Tf,N-1)
    # lin0, = plt.plot(t_vec_u, us[:,0], lw=1)
    # plt.title('Optimal control inputs over time')
    # plt.ylabel("control inputs (N)")
    # plt.xlabel("time (s)")

    return us, xs

