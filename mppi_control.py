import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from pendulum import batched_dynamics, dynamics_rk4


def get_cartpole_mppi_hyperparams():
    """
    Returns a dictionary containing the hyperparameters for running MPPI on the cartpole environment
    The required parameters are:
     * lambda: float parameter between 0. and 1. used to weight samples.
     * Q: torch tensor fo shape (state_size, state_size) representing the state quadratic cost.
     * noise_sigma: torch tensor fo size (action_size, action_size) representing the covariance matrix  of the random action perturbations.
    """
    action_size = 1
    state_size = 6
    hyperparams = {
        'lambda': None,
        'Q': None,
        'noise_sigma': None,
    }
    # --- Your code here

    hyperparams['lambda'] = 0.1
    hyperparams['Q'] = torch.diag(torch.Tensor([8, 20, 20, 1, 0.5, 0.5]))
    hyperparams['noise_sigma'] = torch.Tensor([[20]])

    # ---
    return hyperparams


def get_panda_mppi_hyperparams():
    """
    Returns a dictionary containing the hyperparameters for running MPPI on the panda environment
    The required parameters are:
     * lambda: float parameter between 0. and 1. used to weight samples.
     * Q: torch tensor fo shape (state_size, state_size) representing the state quadratic cost.
     * noise_sigma: torch tensor fo size (action_size, action_size) representing the covariance matrix  of the random action perturbations.
    """
    action_size = 7
    state_size = 14
    hyperparams = {
        'lambda': None,
        'Q': None,
        'noise_sigma': None,
    }
    # --- Your code here

    hyperparams['lambda'] = 0.1
    hyperparams['Q'] = torch.diag(torch.Tensor(
        [10, 10, 10, 10, 10, 10, 10, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
    hyperparams['noise_sigma'] = torch.diag(torch.Tensor(
        [12, 12, 12, 12, 12, 12, 12]))

    # ---
    return hyperparams


class MPPIController(object):

    def __init__(self, env, num_samples, horizon, hyperparams):
        """

        :param env: Simulation environment. Must have an action_space and a state_space.
        :param num_samples: <int> Number of perturbed trajectories to sample
        :param horizon: <int> Number of control steps into the future
        :param hyperparams: <dic> containing the MPPI hyperparameters
        """
        self.env = env
        self.T = horizon
        self.K = num_samples
        self.lambda_ = hyperparams['lambda']
        self.action_size = 1#env.action_space.shape[-1]
        self.state_size = 6#env.state_space.shape[-1]
        self.goal_state = torch.zeros((self.state_size))  # This is just a container for later use
        self.Q = hyperparams['Q'] # Quadratic Cost Matrix (state_size, state_size)
        self.noise_mu = torch.zeros(self.action_size)
        self.noise_sigma = hyperparams['noise_sigma']  # Noise Covariance matrix shape (action_size, action_size)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.U = torch.zeros((self.T, self.action_size)) # nominal action sequence (T, action_size)
        self.u_init = torch.zeros(self.action_size)
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)

    def reset(self):
        """
        Resets the nominal action sequence
        :return:
        """
        self.U = torch.zeros((self.T, self.action_size))# nominal action sequence (T, action_size)

    def command(self, state):
        """
        Run a MPPI step and return the optimal action.
        :param state: torch tensor of shape (state_size,)
        :return:
        """
        action = None
        perturbations = self.noise_dist.sample((self.K, self.T))    # shape (K, T, action_size)
        perturbed_actions = self.U + perturbations      # shape (K, T, action_size)
        trajectory = self._rollout_dynamics(state, actions=perturbed_actions)
        
        trajectory_cost = self._compute_trajectory_cost(trajectory, perturbations)
        self._nominal_trajectory_update(trajectory_cost, perturbations)
        # select optimal action
        action = self.U[0]
        # final update nominal trajectory
        self.U = torch.roll(self.U, -1, dims=0) # move u_t to u_{t-1}
        self.U[-1] = self.u_init # Initialize new end action
        return action

    def _rollout_dynamics(self, state_0, actions):
        """
        Roll out the environment dynamics from state_0 and taking the control actions given by actions
        :param state_0: torch tensor of shape (state_size,)
        :param actions: torch tensor of shape (K, T, action_size)
        :return:
         * trajectory: torch tensor of shape (K, T, state_size) containing the states along the trajectories given by
                       starting at state_0 and taking actions.
                       This tensor contains K trajectories of T length.
         TIP 1: You may need to call the self._dynamics method.
         TIP 2: At most you need only 1 for loop.
        """
        state = state_0.unsqueeze(0).repeat(self.K, 1) # transform it to (K, state_size)
        trajectory = None
        # --- Your code here
        trajectory = torch.zeros((self.K,self.T,self.state_size))
        for t in range(0, self.T):
            state = self._dynamics(state, actions[:,t])
            trajectory[:,t] = state

        # ---
        return trajectory

    def _compute_trajectory_cost(self, trajectory, perturbations):
        """
        Compute the costs for the K different trajectories
        :param trajectory: torch tensor of shape (K, T, state_size)
        :param perturbations: torch tensor of shape (K, T, action_size)
        :return:
         - total_trajectory_cost: torch tensor of shape (K,) containing the total trajectory costs for the K trajectories
        Observations:
        * The trajectory cost be the sum of the state costs and action costs along the trajectories
        * State cost should be quadratic as (state_i-goal_state)^T Q (state_i-goal_state)
        * Action costs should be given by (non_perturbed_action_i)^T noise_sigma^{-1} (perturbation_i)

        TIP 1: the nominal actions (without perturbation) are stored in self.U
        TIP 2: Check Algorithm 2 in https://ieeexplore.ieee.org/document/7989202 for more references.
        """
        total_trajectory_cost = None
        # --- Your code here

        x = trajectory - self.goal_state
        u = self.U
        Q = self.Q
        S = self.noise_sigma_inv

        state_cost = (x*torch.matmul(x, Q)).sum(-1).sum(-1)
        action_cost = (perturbations*torch.matmul(u, S)).sum(-1).sum(-1)
        
        total_trajectory_cost = state_cost + self.lambda_*action_cost
        
        # ---
        return total_trajectory_cost

    def _nominal_trajectory_update(self, trajectory_costs, perturbations):
        """
        Update the nominal action sequence (self.U) given the trajectory costs and perturbations
        :param trajectory_costs: torch tensor of shape (K,)
        :param perturbations: torch tensor of shape (K, T, action_size)
        :return: No return, you just need to update self.U

        TIP: Check Algorithm 2 in https://ieeexplore.ieee.org/document/7989202 for more references about the action update.
        """
        # --- Your code here
        
        beta = torch.min(trajectory_costs)
        gamma = torch.exp((-1/self.lambda_)*(trajectory_costs-beta))
        eta = gamma.sum()
        omega = (1/eta)*gamma
        self.U += torch.matmul(perturbations.permute(1,2,0),omega)
        # ---

    def _dynamics(self, state, action):
        """
        Query the environment dynamics to obtain the next_state in a batched format.
        :param state: torch tensor of size (...., state_size)
        :param action: torch tensor of size (..., action_size)
        :return: next_state: torch tensor of size (..., state_size)
        """
        next_state = batched_dynamics(self.env, state.cpu().detach(), action.cpu().detach())
        #next_state = dynamics_rk4(state.cpu().detach(), action.cpu().detach())
        next_state = torch.tensor(next_state, dtype=state.dtype)
        return next_state
