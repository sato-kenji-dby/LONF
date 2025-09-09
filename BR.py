import numpy as np
import random
# Import Agent class from model module
from model import Agent
# Import global parameters and helper functions from environment module
from environment import cooperation, depent_para, S, F, P, ka, yeta, rho, omega, G, Z, X, W, h, D_T, fun_p, initialize_new_environment_parameters


# The run_BR_simulation function will adapt to the Agent class in the new model
def run_BR_simulation(N_agent, Agent_list, H, M_cooperation, V): # Receives V parameter
    # Loop H times to simulate the BR process
    for round_num in range(H):
        # print(f'BR round: {round_num}')
        
        # Get the current strategy list of all agents
        # all_agents_strategy_list shape is [2, N_agent]
        all_agents_strategy_list = np.array([[agent.d for agent in Agent_list],
                                             [agent.f for agent in Agent_list]])
        
        # Store updated strategies and rewards
        temp_d_list = [0.0] * N_agent
        temp_f_list = [0.0] * N_agent
        temp_reward_list = [0.0] * N_agent
        
        # Iterate through all agents to calculate their best response
        for n in range(N_agent):
            agent = Agent_list[n]
            # Call the Agent instance's own refresh_best_strategy method, passing the V parameter
            new_d, new_f, new_reward = agent.refresh_best_strategy(all_agents_strategy_list, M_cooperation, V)
            temp_d_list[n] = new_d
            temp_f_list[n] = new_f
            temp_reward_list[n] = new_reward
        
        # Update agents' strategies and rewards
        # Here, for simplicity, all agents are updated; random updates can be reintroduced if needed
        for n in range(N_agent):
            Agent_list[n].d = temp_d_list[n]
            Agent_list[n].f = temp_f_list[n]
            Agent_list[n].reward = temp_reward_list[n] # Update the agent's reward attribute

        # Print the average reward for the current round
        avg_reward = sum([agent.reward for agent in Agent_list]) / N_agent
        # print(f'Average reward in BR round {round_num}: {avg_reward}')

    # Print final strategies (d and f)
    final_d_strategy = [agent.d for agent in Agent_list]
    final_f_strategy = [agent.f for agent in Agent_list]
    # print(f'Final d strategies: {final_d_strategy}')
    # print(f'Final f strategies: {final_f_strategy}')
