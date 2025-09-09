import json, os
import copy
import numpy as np
import matplotlib.pyplot as plt
import random # Ensure random is imported, as Agent initialization might use it

np.random.seed(42)

# Import run_BR_simulation from BR.py (modified for the new model)
from BR import run_BR_simulation
# Import run_masl_simulation from masl.py
from masl import run_masl_simulation
# Import run_global_optimization from global_optimizer.py
from global_optimizer import run_global_optimization
# Import global parameters for the new model from environment.py
# initialize_new_environment_parameters is now automatically executed when environment is imported
from environment import S, F, B, P, ka, tao, yeta, rho, gama, omega, D_T, G, Z, X, W, h, fun_p, M_cooperation_global
# Import Agent class from model.py (modified for the new model)
from model import Agent

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Note: This Potential function might no longer be directly used for Q and utility calculations in the new model,
# but can be kept if potential function visualization is needed.
def Potential(strategy_list, M_cooperation, num_agents):
    # Ensure S, F, B, P, ka, tao, yeta, rho, gama, omega, D_T, G, Z, X, W, h are imported and initialized from environment
    # These are global variables and can be used directly

    d = strategy_list[0]
    f = strategy_list[1]
    
    sum_profit = fun_p(d, S, G)
    
    sum_cooperation = 0
    for i in range(num_agents): # num_agents should be passed externally
        sum_cooperation += (d[i]*S[0][i]*np.sum(M_cooperation[i])-np.sum(d*S*M_cooperation[i]) + \
                            f[i]*F[0][i]*np.sum(M_cooperation[i])-np.sum(f*F*M_cooperation[i])) / Z[i][0]

    sum_cooperation *= gama
    sum_energy = np.sum(W*(f*f)*d/Z.T)
    U = sum_profit - sum_energy  + sum_cooperation
    
    # Check time condition
    # D_T[0] is a row vector, yeta[0] * d * S[0] / (f * F[0]) is also a row vector
    # Ensure all elements meet the condition
    if np.any(D_T[0] < (yeta[0] * d * S[0]) / (f * F[0])):
        # According to user feedback, this is no longer a hard penalty, but rather Q accumulates deficits.
        # So the return value of the Potential function may no longer be -100.
        # This is temporarily kept; if needed, it can be adjusted according to the actual model definition.
        pass 
    return U

if __name__ == '__main__':
    # --- Algorithm Selection ---
    # Set to 'MASL' to run MASL, 'BR' to run BR, 'GLOBAL' to run global optimization
    ALGORITHM_CHOICE = 'BR' 
    
    # New model parameter initialization
    num_agents = 5 # Number of agents, corresponding to num in BR__.py
    max_rounds = 10 # Number of optimization iterations (BR, MASL, or GLOBAL)
    num_timeslots = 100 # Number of time slots, corresponding to T in the old model
    learning_rate = 0.0001 # Learning rate for MASL

    # M_cooperation is now imported as a global variable from environment

    # Initialize agent list
    # The Agent class now requires agent_id, num_agents, initial_d, initial_f
    # Initial strategies can be randomly generated
    initial_agent_list = [Agent(agent_id=i, num_agents=num_agents, 
                                initial_d=random.uniform(0, 1), 
                                initial_f=random.uniform(0, 1)) 
                          for i in range(num_agents)]

    Q_save = [] # Stores the average Q over all time slots for each V_temp
    utility_save = [] # Stores the average utility over all time slots for each V_temp

    # V_list remains unchanged, used for parameter sensitivity analysis
    V_list = []
    V_list += list(range(10, 20100, 5000))
    V_list += list(range(20000, 60100, 10000))

    V_list = [10000]

    for V_temp in V_list: # Outer loop for V sensitivity analysis
        # Reset agent state at the beginning of each V_temp loop
        Agent_list = copy.deepcopy(initial_agent_list)
        
        Q_per_V = [] # Stores Q for each time slot under current V_temp
        utility_per_V = [] # Stores utility for each time slot under current V_temp

        for t_slot in range(num_timeslots): # Time slot loop (multiple rounds of actions)
            # print(f'\nTime Slot: {t_slot}')

            if False: # This block is for testing and should be removed or commented out
                # print(f"All agents' Q are too high, skipping optimization and forcing d=0, f=1 for all agents.")
                for agent in Agent_list:
                    agent.d = 0.0
                    agent.f = 1.0
            else:
                # Run algorithm based on choice
                if ALGORITHM_CHOICE == 'MASL':
                    # Run MASL optimization process
                    run_masl_simulation(num_agents, Agent_list, max_rounds, learning_rate, M_cooperation_global, V_temp)
                elif ALGORITHM_CHOICE == 'BR':
                    # Run BR optimization process
                    run_BR_simulation(num_agents, Agent_list, max_rounds, M_cooperation_global, V_temp)
                elif ALGORITHM_CHOICE == 'GLOBAL':
                    # Run global optimization process
                    run_global_optimization(num_agents, Agent_list, max_rounds, M_cooperation_global, V_temp)
                else:
                    raise ValueError("Invalid ALGORITHM_CHOICE specified.")
            
            # After optimization, calculate the total reward (utility) for all agents in the current time slot
            # and update each agent's Q (time constraint violation deficit)
            total_reward_current_timeslot = 0
            total_Q_current_timeslot = 0
            Q_current_timeslot = []
            sum_time_current_timeslot = []
            for agent in Agent_list:
                # After BR optimization, Agent.d and Agent.f are already optimized strategies.
                # Need to recalculate reward and current_sum_time once to update Q.
                # Pass the current strategies of all agents in Agent_list and V_temp.
                current_strategies_for_reward_calc = np.array([[a.d for a in Agent_list],
                                                                [a.f for a in Agent_list]])
                agent.calculate_reward(current_strategies_for_reward_calc, M_cooperation_global, V_temp) # Ensure reward and current_sum_time are up-to-date
                agent.refresh_Q() # Update Q
                
                total_reward_current_timeslot += agent.reward
                total_Q_current_timeslot += agent.Q # Accumulate Q
                Q_current_timeslot.append(agent.Q)
                sum_time_current_timeslot.append(agent.sum_time)

            utility_per_V.append(total_reward_current_timeslot / num_agents) # Average utility
            Q_per_V.append(total_Q_current_timeslot / num_agents) # Average Q

            # print(f'Time Slot {t_slot} - Average Utility: {utility_per_V[-1]}')
            # print(f'Time Slot {t_slot} - Average Q: {Q_per_V[-1]}')
            # print(f'Time Slot {t_slot} - Q: {Q_current_timeslot}')
            # print(f'Time Slot {t_slot} - Training Time: {sum_time_current_timeslot}')

        # Store average Q and average utility for the current V_temp
        Q_save.append(sum(Q_per_V) / len(Q_per_V))
        utility_save.append(sum(utility_per_V) / len(utility_per_V))

    # --- Save Results ---
    result_directory = f'./results'
    ensure_directory(result_directory)

    with open(f'./results/results-Q.json', 'w') as f:
        json.dump(Q_save, f)
    with open(f'./results/results-utility.json', 'w') as f:
        json.dump(utility_save, f)
    with open(f'./results/results-V.json', 'w') as f:
        json.dump(V_list, f)

    # --- Plotting ---
    fig, ax1 = plt.subplots()
    # Plot the first line graph (Q)
    ax1.plot(V_list, Q_save, color='blue', label='Q (Time Violation Deficit)')
    ax1.set_xlabel('V (Parameter)')
    ax1.set_ylabel('Q', color='blue')
    ax1.tick_params('y', colors='blue')
    # Add a second y-axis, sharing the x-axis (Utility)
    ax2 = ax1.twinx()
    ax2.plot(V_list, utility_save, color='red', label='Utility (Total Reward)')
    ax2.set_ylabel('Utility', color='red')
    ax2.tick_params('y', colors='red')
    # Add legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    # Show plot
    plt.title('Q and Utility vs. V')
    plt.show()
