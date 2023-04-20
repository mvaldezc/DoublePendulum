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
    # running_cost = 0.5*(curr_x - xstar).T @ Q @ (curr_x - xstar) + 0.5 * curr_u * R * curr_u
    running_cost = 0.5*(curr_x - xstar).T @ Q @ (curr_x - xstar) + 0.5 * curr_u.T @ R @ curr_u
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

def total_cost_reduction(alpha, ks, qus, quus):
    delta_J = alpha * ks @ qus + (alpha**2)/2 * ks @ quus @ ks
    delta_J = torch.sum(delta_J, dim=0).squeeze(-1)
    return delta_J

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
def backward_pass(xs_nom, us_nom, xstar, Q, R, Qf, mu, mu_delta_0, mu_min, mu_delta):

    # get the number of timesteos
    N = xs_nom.shape[0]

    # create a tensor for storing all Vx (jacobian of value function) and Vxx (hessian of the value function) values
    Vxs = torch.zeros((N, 6))
    Vxxs = torch.zeros((N, 6, 6))

    # Set Vxs(N) = Qf * (xs(N) - xstar) and Vxxs(N) = Qf
    Vxs[N-1, :] = (Qf @ (xs_nom[N-1, :] - xstar).reshape(6,1)).squeeze(-1)
    Vxxs[N-1, :, :] = Qf

    # create a tensor for storing all the gains k and K for the entire trajectory
    ks = torch.ones(N-1, 1, 1) # TODO: initialize to zero?
    Ks = torch.zeros(N-1, 1, 6) # TODO: check the dimensions of these

    # create a tensor that stores all of the qus and quus
    qus = torch.zeros(N-1, 1, 1)
    quus = torch.zeros(N-1, 1, 1)

    # for each value in the nominal trajectory
    restart_bck_pass = False
    for t in reversed(range(1,N,1)): # t goes from N-1 to 1, t-1 goes from N-2 to 0

        # get the current state and control from the nominal trajectory (t-1)
        curr_x = xs_nom[t-1, :].reshape(6,1)
        curr_u = us_nom[t-1, :].reshape(1,1)

        # linearize about the current point in the nominal trajectory (t-1)
        A, B = linearize_dynamics(curr_x.reshape(6,), curr_u.reshape(1,))  

        # compute the first and second derivative of the lienarized dynamics (lx, lu, etc) (t-1)
        lx = ((curr_x - xstar.reshape(6,1)).T @ Q.T).reshape(6,1)
        lu = ((curr_u).T @ R.T).reshape(1,1)
        lxx = Q
        luu = R
        lux = 0

        # obtain the first and second derivatives of the value function (Vx') for t
        Vx = Vxs[t, :].reshape(6,1)
        Vxx = Vxxs[t, :, :].squeeze(0) # (6,6)

        # compute the first derivatives qx and qu (t-1)
        qx, qu = compute_qx_qu(lx, A, Vx, lu, B)

        # compute the second derivatives Qxx, Quu, Qux (t-1)
        qxx, quu, qux = compute_qxx_quu(lxx, A, Vxx, luu, B, lux, mu)

        # store all of the qus and quus
        qus[t-1, :, :] = qu
        quus[t-1, :, :] = quu

        if quu <= 0: # not positive definite, increase mu
            mu_delta = max(mu_delta_0, mu_delta*mu_delta_0)
            mu = max(mu_min, mu*mu_delta)
            restart_bck_pass = True
            break

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
        Vxs[t-1, :] = Vx.squeeze(-1)
        Vxxs[t-1, :] = Vxx

    return ks, Ks, qus, quus, restart_bck_pass, mu, mu_delta

    # set up the forward pass function to obtain the new xs and us
def forward_pass(ks, Ks, N, xs_nom, us_nom, alpha):
    # save a tensor of all the new states and control inputs
    us = torch.zeros((N-1, 1))
    xs = torch.zeros((N, 6))

    # set the first stae of the new xs equal to the first state of the nominal x
    xs[0, :] = xs_nom[0, :]

    # iterate through all the timepoints
    for t in range(N-1): # t goes from 0 to N-2
        # get the nominal u, nominal x, and current x for the given timstep
        curr_nom_u = us_nom[t, :].reshape(1,1)
        curr_nom_x = xs_nom[t, :].reshape(6,1)
        curr_x = xs[t, :].reshape(6,1)

        # get the current gains for the given timestep
        k = ks[t, :].reshape(1,1)
        K = Ks[t, :].reshape(6,1)

        # calculate delta x from the current x and the nominal x
        delta_x = curr_x - curr_nom_x 
        # calculate delta u using the gains and delta x
        delta_u = alpha*k + torch.dot(K.squeeze(-1), delta_x.squeeze(-1))
        # calculate the new u from delta u
        curr_u = curr_nom_u + delta_u # (1,1)
        # obtain the next state from the dynamics function
        next_x = dynamics_rk4(curr_x.reshape(1,6), curr_u) # (1,6)

        # save the current u and the next x to the stored vectors of us and xs
        us[t, :] = curr_u
        xs[t+1, :] = next_x

    return us, xs

## perform iLQR
def run_ilqr(current_state, N, Tf, num_iterations, xstar, mu, mu_delta_0, mu_delta, mu_min, Q, R, Qf, alpha, c, gamma, eps):

    # perform the rollout to obtain the nominal trajectory
    xs_nom, us_nom = obtain_nominal_trajectory(current_state, N, Tf)

    # run loop until convergence is reached
    i = 0
    xs = xs_nom
    us = us_nom

    mu = mu_min
    mu_delta = mu_delta_0
    cost_diff = 2*eps
    # while i < num_iterations:
    # test for convergence: if the absolute value of the difference in costs is greater than a threshold, 
    # move onto the next iteration of mpc because the control input is good enough
    #while cost_diff > eps:
    while i < num_iterations:
        #print(f'{i=}')
        # perform the backwards pass
        restart_bck_pass = True
        while restart_bck_pass == True:
            #print('restarting bck pass')
            ks, Ks, qus, quus, restart_bck_pass, mu, mu_delta = backward_pass(
                xs, us, xstar, Q, R, Qf, mu, mu_delta_0, mu_min, mu_delta)
        # after the backward pass ends, decrease mu
        mu_delta = min(1/mu_delta_0, mu_delta/mu_delta_0)
        if mu*mu_delta > mu_min:
            mu = mu*mu_delta
        else:
            mu = 0

        cost_reduction_sufficient = False
        alpha = 1.0
        J_prev = total_cost_func(xs, us, xstar, Q, R, Qf)
        while cost_reduction_sufficient == False:
            #print('restarting fwd pass')
            # perform the forward pass
            us_test, xs_test = forward_pass(ks, Ks, N, xs, us, alpha)

            # calculate the cost for the current state
            #print(f'{J_prev=}')
            J_curr = (total_cost_func(xs_test, us_test, xstar, Q, R, Qf))
            #print(f'{J_curr=}')

            # calculate the difference in cost for the previous and current state
            delta_J = J_prev - J_curr
            #print(f'{delta_J=}')

            # calculate the expected total cost reduction
            expected_delta_J = -total_cost_reduction(alpha, ks, qus, quus)
            #print(f'{expected_delta_J=}')

            # caclulate the delta_J threshold
            z = delta_J/expected_delta_J
            #print(f'{z=}')

            if z > c:
            #if z >= 0:
                cost_reduction_sufficient = True
            # if the cost reduction is not sufficient, reduce alpha
            else:
                alpha = alpha*gamma
                #rint(f'{alpha=}')

                if alpha < 1e-6:
                    #print('alpha is small')
                    break

            cost_diff = abs(J_prev - J_curr)
            #print(f'{cost_diff=}')

        us = us_test
        xs = xs_test
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

