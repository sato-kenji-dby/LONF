import numpy as np
from model import Agent

def run_masl_simulation(N_agent, Agent_list, H, Lr, M_cooperation, V):
    """
    Runs the simulation using a multi-agent gradient ascent algorithm.

    Args:
        N_agent (int): Number of agents.
        Agent_list (list): List of Agent objects.
        H (int): Number of simulation iterations.
        Lr (float): Learning rate.
        M_cooperation (np.ndarray): Cooperation matrix.
        V (float): V parameter in the Lyapunov function.
    """
    epsilon = 1e-5  # Small value for numerical gradient calculation

    for round_num in range(H):
        # print(f'MASL round: {round_num}')

        # Get the current strategy list of all agents
        all_agents_strategy_list = np.array([[agent.d for agent in Agent_list],
                                             [agent.f for agent in Agent_list]])

        # Store gradients for each agent's strategy
        gradients_d = np.zeros(N_agent)
        gradients_f = np.zeros(N_agent)

        # Calculate gradients for each agent
        for n in range(N_agent):
            agent = Agent_list[n]

            # Calculate current reward
            base_reward = agent.calculate_reward(all_agents_strategy_list, M_cooperation, V)

            # Calculate gradient for d
            d_perturbed_strategy = all_agents_strategy_list.copy()
            d_perturbed_strategy[0, n] += epsilon
            reward_d_perturbed = agent.calculate_reward(d_perturbed_strategy, M_cooperation, V)
            gradients_d[n] = (reward_d_perturbed - base_reward) / epsilon

            # Calculate gradient for f
            f_perturbed_strategy = all_agents_strategy_list.copy()
            f_perturbed_strategy[1, n] += epsilon
            reward_f_perturbed = agent.calculate_reward(f_perturbed_strategy, M_cooperation, V)
            gradients_f[n] = (reward_f_perturbed - base_reward) / epsilon

        # Update strategies for all agents
        for n in range(N_agent):
            agent = Agent_list[n]
            # Update strategies using gradient ascent
            new_d = agent.d + Lr * gradients_d[n]
            new_f = agent.f + Lr * gradients_f[n]

            # Clip strategies to the range [0.001, 1]
            agent.d = np.clip(new_d, 1e-3, 1.0)
            agent.f = np.clip(new_f, 1e-3, 1.0)

        # Add print statements here to monitor the learning process if needed
        avg_reward = sum([agent.calculate_reward(np.array([[a.d for a in Agent_list], [a.f for a in Agent_list]]), M_cooperation, V) for agent in Agent_list]) / N_agent
        # print(f'Average reward in MASL round {round_num}: {avg_reward}')

    # Print final strategies
    final_d_strategy = [agent.d for agent in Agent_list]
    final_f_strategy = [agent.f for agent in Agent_list]
    # print(f'Final d strategies: {final_d_strategy}')
    # print(f'Final f strategies: {final_f_strategy}')
