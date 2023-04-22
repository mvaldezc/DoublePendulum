## ROB 498 Winter '23 Final Project
# Marco Antonio Valdez Calderon
# Marcela De los Rios 

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from numpngw import write_apng
from IPython.display import Image
from IPython.display import display
from pendulum import *
import mujoco
from lqr import *
from mppi_control import MPPIController, get_cartpole_mppi_hyperparams
import time # TODO: remove

def demo_ilqr(pos, vel, acc, num_mpc_iterations, N, Th, num_lqr_iterations, xstar,
                         mu, mu_delta_0, mu_delta, mu_min, Q, R, Qf, alpha, c, gamma, eps, alpha_min):
    print('executing ilqr demo')
    env = gym.make('InvertedDoublePendulum-v4', render_mode='rgb_array')
    start_state = set_state(env, pos, vel, acc)
    current_state = change_of_coords(start_state)
    #_, _, _, _, _ = env.step(np.array([0]))

    # store all of the actions over all the iterations of LQR
    actions = torch.zeros((num_mpc_iterations, 1))
    states = torch.zeros((num_mpc_iterations+1, 6))
    states[0, :] = current_state

    frames = []  # frames to create animated png
    frames.append(env.render())

    # for i in tqdm(range(num_mpc_iterations)):
    for i in range(num_mpc_iterations):
        
        # run the iLQR optimization
        us, _ = run_ilqr(current_state, N, Th, num_lqr_iterations, xstar,
                         mu, mu_delta_0, mu_delta, mu_min, Q, R, Qf, alpha, c, gamma, eps, alpha_min)
        # get the first control input value from the ilqr optimization
        action = us[0, :]

        # apply the action to move the simulation to the next state
        next_state, _, _, _, _ = env.step(action)
        next_state = change_of_coords(next_state)
        
        img = env.render()
        frames.append(img)
        # store all of the actions and the states that were taken
        actions[i, :] = action
        states[i+1, :] = next_state
        current_state = next_state

    return states, actions, frames

def plot_multiple_states(labels_list, linestyle_list, color_list, states_list, actions_list, num_mpc_iterations, Tf, init_state_type):
    print('plotting multiple states')
    
    # plot the optimal state trajectory over time
    plt.figure()
    t_vec_x = np.linspace(0, Tf, num_mpc_iterations+1)
    fig, axs = plt.subplots(3, 1, sharex=True)

    for i in range(len(states_list)):
        lin0, = axs[0].plot(t_vec_x, states_list[i][:, 0], linestyle_list[i], color=color_list[i], lw=2, label=labels_list[i])
        lin1, = axs[1].plot(t_vec_x, states_list[i][:, 1], linestyle_list[i], color=color_list[i], lw=2)
        lin2, = axs[2].plot(t_vec_x, states_list[i][:, 2], linestyle_list[i], color=color_list[i], lw=2)
    axs[0].set_title(f'Optimal position trajectories over time: {init_state_type}')
    axs[0].set_ylabel(r'$x_{cart}$ (m)')
    axs[1].set_ylabel(r'$\theta_1$ (rad)')
    axs[2].set_ylabel(r'$\theta_2$ (rad)')
    axs[-1].set_xlabel("Time (s)")
    axs[0].legend()

    # plot the optimal state trajectory over time
    plt.figure()
    t_vec_x = np.linspace(0, Tf, num_mpc_iterations+1)
    fig, axs = plt.subplots(3, 1, sharex=True)

    for i in range(len(states_list)):
        lin0, = axs[0].plot(t_vec_x, states_list[i][:, 3], linestyle_list[i], color=color_list[i], lw=2, label=labels_list[i])
        lin1, = axs[1].plot(t_vec_x, states_list[i][:, 4], linestyle_list[i], color=color_list[i], lw=2)
        lin2, = axs[2].plot(t_vec_x, states_list[i][:, 5], linestyle_list[i], color=color_list[i], lw=2)
    axs[0].set_title(f'Optimal velocity trajectories over time: {init_state_type}')
    axs[0].set_ylabel(r'$\dot{x}_{cart}$ (m/s)')
    axs[1].set_ylabel(r'$\dot{\theta}_1$ (rad/s)')
    axs[2].set_ylabel(r'$\dot{\theta}_2$ (rad/s)')
    axs[-1].set_xlabel("Time (s)")
    axs[0].legend()

    # plot the optimal control trajectory
    plt.figure()
    t_vec_u = np.linspace(0, Tf, num_mpc_iterations)
    for i in range(len(actions_list)):
        lin0, = plt.plot(t_vec_u, actions_list[i][:, 0], linestyle_list[i], color=color_list[i], lw=1, label=labels_list[i])
    plt.title(f'Optimal control inputs over time: {init_state_type}')
    plt.ylabel("control inputs (N)")
    plt.xlabel("Time (s)")
    plt.legend()

def demo_mppi(pos,vel,acc,mppi_iter):
    print('executing mppi demo')

    ## run MPC on the simulated double inverted pendulum
    # initialize the inverted pedulum object
    env = gym.make('InvertedDoublePendulum-v4', render_mode='rgb_array')
    start_state = set_state(env, pos, vel, acc)
    current_state = change_of_coords(start_state)
    #current_state, _, _, _, _ = env.step(np.array([0]))
    #current_state = change_of_coords(current_state)

    goal_state = np.zeros(6)
    controller = MPPIController(env, num_samples=100, horizon=30, hyperparams=get_cartpole_mppi_hyperparams())
    controller.goal_state = torch.tensor(goal_state, dtype=torch.float32)

    # store all of the actions over all the iterations of MPPI
    actions = torch.zeros((mppi_iter, 1))
    states = torch.zeros((mppi_iter+1, 6))
    states[0, :] = current_state

    frames = []  # frames to create animated png
    frames.append(env.render())

    # pbar = tqdm(range(mppi_iter))
    # for i in pbar:
    for i in range(mppi_iter):
        
        # get the first control input value from the MPPI optimization
        action = controller.command(current_state) # returns us[0,:]

        # apply the action to move the simulation to the next state
        next_state, _, _, _, _ = env.step(action)
        next_state = change_of_coords(next_state)
        error_i = np.linalg.norm(next_state-goal_state)
        # pbar.set_description(f'Goal Error: {error_i:.4f}')
        img = env.render()
        frames.append(img)
        # store all of the actions and the states that were taken
        actions[i, :] = action
        states[i+1, :] = next_state
        current_state = next_state

        if error_i < .2:
            break

    return states, actions, frames

def run_demo():
    print('obtaining simulation parameters')
    # obtain simulation parameters
    Th = 0.9 # this refers to the time horizon for the optimization
    dt = 0.05
    N = int(Th/dt + 1)
    num_lqr_iterations = 1
    mu_min = 1e-6
    mu = mu_min
    mu_delta_0 = 2
    mu_delta = mu_delta_0
    Tf = 3 #5 # total time running mpc
    num_mpc_iterations = int(Tf/dt + 1)
    # for backtracking line search
    gamma = 0.5
    alpha = 1.0
    c = 1e-4 # inner loop cost tolerance
    eps = 1e-6 # outer loop cost tolerance
    alpha_min = 5e-2

    # LQR parameters
    # set the desired state (xstar), should be the goal state
    # will will be linearizing about this point
    xstar = torch.zeros((6,))
    # set the Q and R matrices
    Q = torch.diag(torch.tensor([50.0, 65.0, 65.0, 1.0, 22.0, 22.0]))
    R = torch.tensor([[1.0]])
    Qf = 6.0 * Q

    mppi_iter = 61

    pos = [0, -0.1, 0.1]
    vel = [0, 0, 0]
    x0 = torch.cat((torch.tensor(pos), torch.tensor(vel))).reshape(1,6)
    acc = dynamics_accel(x0, torch.tensor([[0.0]])).reshape(3,)
    acc = acc.tolist()

    states1_ilqr, actions1_ilqr, frames1_ilqr = demo_ilqr(pos, vel, acc, num_mpc_iterations, N, Th, num_lqr_iterations, xstar,
                         mu, mu_delta_0, mu_delta, mu_min, Q, R, Qf, alpha, c, gamma, eps, alpha_min)
    states1_mppi, actions1_mppi, frames1_mppi = demo_mppi(pos, vel, acc, mppi_iter)

    pos = [0, 0, 0]
    vel = [0, -0.2, 0]
    x0 = torch.cat((torch.tensor(pos), torch.tensor(vel))).reshape(1, 6)
    acc = dynamics_accel(x0, torch.tensor([[0.0]])).reshape(3,)
    acc = acc.tolist()

    states2_ilqr, actions2_ilqr, frames2_ilqr = demo_ilqr(pos, vel, acc, num_mpc_iterations, N, Th, num_lqr_iterations, xstar,
                         mu, mu_delta_0, mu_delta, mu_min, Q, R, Qf, alpha, c, gamma, eps, alpha_min)
    states2_mppi, actions2_mppi, frames2_mppi = demo_mppi(pos, vel, acc, mppi_iter)

    pos = [0, np.pi, 0]
    vel = [0, 0, 0]
    x0 = torch.cat((torch.tensor(pos), torch.tensor(vel))).reshape(1, 6)
    acc = dynamics_accel(x0, torch.tensor([[0.0]])).reshape(3,)
    acc = acc.tolist()

    states3_ilqr, actions3_ilqr, frames3_ilqr = demo_ilqr(pos, vel, acc, num_mpc_iterations, N, Th, num_lqr_iterations, xstar,
                         mu, mu_delta_0, mu_delta, mu_min, Q, R, Qf, alpha, c, gamma, eps, alpha_min)
    states3_mppi, actions3_mppi, frames3_mppi = demo_mppi(pos, vel, acc, mppi_iter)

    pos = [-0.3, np.pi/2, -np.pi/2]
    vel = [0, -1.0, 1.0]
    x0 = torch.cat((torch.tensor(pos), torch.tensor(vel))).reshape(1, 6)
    acc = dynamics_accel(x0, torch.tensor([[0.0]])).reshape(3,)
    acc = acc.tolist()

    states4_ilqr, actions4_ilqr, frames4_ilqr = demo_ilqr(pos, vel, acc, num_mpc_iterations, N, Th, num_lqr_iterations, xstar,
                         mu, mu_delta_0, mu_delta, mu_min, Q, R, Qf, alpha, c, gamma, eps, alpha_min)
    states4_mppi, actions4_mppi, frames4_mppi = demo_mppi(pos, vel, acc, mppi_iter)

    write_apng("pend_ilqr1.png", frames1_ilqr, delay=5)
    write_apng("pend_mppi1.png", frames1_mppi, delay=5)
    write_apng("pend_ilqr2.png", frames2_ilqr, delay=5)
    write_apng("pend_mppi2.png", frames2_mppi, delay=5)
    write_apng("pend_ilqr3.png", frames3_ilqr, delay=5)
    write_apng("pend_mppi3.png", frames3_mppi, delay=5)
    write_apng("pend_ilqr4.png", frames4_ilqr, delay=5)
    write_apng("pend_mppi4.png", frames4_mppi, delay=5)
    # print('Shown belown: simulations of the double inverted pendulum on a cart for iLQR an MPPI on different initial states')
    # print('Simulation of iLQR on initial state 1:')
    display(Image(filename="pend_ilqr1.png"))
    # print('Simulation of MPPI on initial state 1:')
    display(Image(filename="pend_mppi1.png"))
    # print('Simulation of iLQR on initial state 2:')
    display(Image(filename="pend_ilqr2.png"))
    # print('Simulation of MPPI on initial state 2:')
    display(Image(filename="pend_mppi2.png"))
    # print('Simulation of iLQR on initial state 3:')
    display(Image(filename="pend_ilqr3.png"))
    # print('Simulation of MPPI on initial state 3:')
    display(Image(filename="pend_mppi3.png"))
    # print('Simulation of iLQR on initial state 4:')
    display(Image(filename="pend_ilqr4.png"))
    # print('Simulation of MPPI on initial state 4:')
    display(Image(filename="pend_mppi4.png"))

    init_state_type = 'initial-up states'
    # print('Shown below: position, velocity, and control input trajectories for initial states 1 and 2 (initial-up configuration) for iLQR and MPPI ')
    plot_multiple_states([r'$x(0)_1$ iLQR',r'$x(0)_2$ iLQR',r'$x(0)_1$ MPPI',r'$x(0)_2$ MPPI'],["-","-",":",":"],
                        ["b", "r", "b", "r"], [states1_ilqr, states2_ilqr, states1_mppi, states2_mppi], [actions1_ilqr, actions2_ilqr, actions1_mppi, actions2_mppi], num_mpc_iterations, Tf, init_state_type)
    init_state_type = 'initial-down states'
    # print('Shown below: position, velocity, and control input trajectories for initial states 3 and 4 (initial-down configuration) for iLQR and MPPI ')
    plot_multiple_states([r'$x(0)_3$ iLQR', r'$x(0)_4$ iLQR', r'$x(0)_3$ MPPI', r'$x(0)_4$ MPPI'],
                        ["-", "-", ":", ":"], ["b", "r", "b", "r"], [states3_ilqr, states4_ilqr, states3_mppi, states4_mppi], [actions3_ilqr, actions4_ilqr, actions3_mppi, actions4_mppi], num_mpc_iterations, Tf, init_state_type)
    plt.show()

if __name__ == '__main__':
    t = time.time()
    run_demo()
    elapsed_time = time.time() - t
    print(f'Demo completed, {elapsed_time=}')