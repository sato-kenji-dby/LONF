import numpy as np
import scipy.optimize as opt
from model import Agent
from environment import S, F, B, P, ka, tao, yeta, rho, gama, omega, D_T, G, Z, X, W, h, fun_p, M_cooperation_global

def _objective_function_for_global_optimization(x, N_agent, Agent_list_template, M_cooperation, V):
    """
    Objective function for global optimization.
    x is a flattened array containing d and f strategies for all agents.
    Agent_list_template is a copy of the list of Agent objects, used to calculate rewards.
    """
    d_strategies = x[:N_agent]
    f_strategies = x[N_agent:]

    # Clip strategies to [0.001, 1] to prevent the optimizer from exploring invalid regions
    d_strategies = np.clip(d_strategies, 1e-3, 1.0)
    f_strategies = np.clip(f_strategies, 1e-3, 1.0)

    # Construct the strategy list for all agents
    all_agents_strategy_list = np.array([d_strategies, f_strategies])

    total_reward = 0
    for n in range(N_agent):
        agent = Agent_list_template[n]
        # When calculating the reward, use the current agent's Q value
        # Note: Here, it is necessary to ensure that the Q values of the Agent objects in Agent_list_template are correct.
        # In main.py, Agent_list is copied at the beginning of each time slot, so the Q values are for the current time slot.
        reward = agent.calculate_reward(all_agents_strategy_list, M_cooperation, V)
        total_reward += reward
    
    return -total_reward # Return negative value because the optimizer seeks to minimize

def run_global_optimization(N_agent, Agent_list, H, M_cooperation, V):
    """
    Runs the simulation using a global optimization algorithm.

    Args:
        N_agent (int): Number of agents.
        Agent_list (list): List of Agent objects.
        H (int): Number of optimization iterations (for global optimization, this is usually maxiter of the optimizer).
        M_cooperation (np.ndarray): Cooperation matrix.
        V (float): V parameter in the Lyapunov function.
    """
    # print(f'Running Global Optimization for {H} iterations.')

    # Initial guess: use strategies from the current Agent_list
    initial_d = np.array([agent.d for agent in Agent_list])
    initial_f = np.array([agent.f for agent in Agent_list])
    x0 = np.concatenate((initial_d, initial_f))

    # Define bounds for d and f
    bounds = [(1e-3, 1.0)] * (2 * N_agent) # d and f are both between [0.001, 1]

    # Use dual_annealing for quick example
    # maxiter corresponds to H
    res = opt.dual_annealing(_objective_function_for_global_optimization, bounds, 
                             args=(N_agent, Agent_list, M_cooperation, V),
                             maxiter=H,
                             seed=42) # Add random seed for reproducibility

    optimized_x = res.x
    optimized_d = optimized_x[:N_agent]
    optimized_f = optimized_x[N_agent:]
    max_total_reward = -res.fun # returns the negative of the minimum, so take the negative

    # Update strategies in Agent_list
    for n in range(N_agent):
        Agent_list[n].d = optimized_d[n]
        Agent_list[n].f = optimized_f[n]
        # Recalculate each agent's reward to ensure its reward attribute is up-to-date
        # Note: The all_agents_strategy_list passed here should be the optimized global strategy
        current_strategies_for_reward_calc = np.array([optimized_d, optimized_f])
        Agent_list[n].reward = Agent_list[n].calculate_reward(current_strategies_for_reward_calc, M_cooperation, V)

    # print(f'Global Optimization finished. Max Total Reward: {max_total_reward}')
    # print(f'Final d strategies (Global Opt): {optimized_d}')
    # print(f'Final f strategies (Global Opt): {optimized_f}')
