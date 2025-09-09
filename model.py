import numpy as np
import random
import scipy.optimize as opt # Import scipy.optimize
from math import floor # Import floor

# Import global parameters for the new model from environment module
from environment import S, F, B, P, ka, tao, yeta, rho, gama, omega, D_T, G, Z, X, W, h, fun_p

def discretize_and_round_to_right_edge(data: np.ndarray, num_segments: int) -> np.ndarray:
    """
    Discretizes a continuous value [0-1] and rounds it to the right edge of each segment.

    Args:
        data: Input NumPy 1D array with values between [0, 1].
        num_segments: Number of segments to divide the [0, 1] interval into.

    Returns:
        A NumPy 1D array containing the discretized and rounded values.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 1:
        raise ValueError("Input data must be a NumPy 1D array")
    if not np.all((data >= 0) & (data <= 1)):
        raise ValueError("Input data values must be between [0, 1]")
    if not isinstance(num_segments, int) or num_segments <= 0:
        raise ValueError("Number of segments must be a positive integer")

    # Calculate the width of each segment
    segment_width = 1.0 / num_segments

    # Calculate the "segment value" (1-based) for each value, i.e., to which segment's right edge it should be rounded.
    # np.ceil(data * num_segments) maps [0, 1] to [0, num_segments] and rounds up.
    # Example: 0.1 -> 0.4 -> ceil(0.4) = 1 (right edge of the first segment)
    #          0.25 -> 1.0 -> ceil(1.0) = 1 (right edge of the first segment)
    #          0.2500001 -> 1.0000004 -> ceil(1.0000004) = 2 (right edge of the second segment)
    segment_values = np.ceil(data * num_segments)

    # Special handling: If the input value is 0, np.ceil(0 * num_segments) will be 0.
    # We want 0 to fall on the right edge of the first segment, so set its segment_value to 1.
    # All values exactly equal to 0 should be rounded to the right boundary of the first segment.
    segment_values[data == 0] = 1

    # Multiply the segment value by the segment width to get the final rounded value
    # Example: Segment value 1 * 0.25 = 0.25
    #          Segment value 2 * 0.25 = 0.5
    rounded_values = segment_values * segment_width

    return rounded_values


class Agent:
    def __init__(self, agent_id, num_agents, initial_d=None, initial_f=None):
        self.agent_id = agent_id # Used to identify itself in the strategy list
        # Strategy components d and f in the new model
        # Initial values can be randomly generated or chosen from a strategy_pool
        self.d = initial_d if initial_d is not None else random.uniform(0, 1) # Assume d is between [0,1]
        self.f = initial_f if initial_f is not None else random.uniform(0, 1) # Assume f is between [0,1]
        
        # Cost/reward required for decision-making (reward in the new model)
        self.reward = 0 # Corresponds to the return value of fun_C

        # New: Time constraint violation deficit and accumulated queue
        self.time_violation_deficit = 0 # Time constraint violation deficit for the current time slot (if sum_time < 0)
        self.current_sum_time = 0 # Value of sum_time for the current time slot
        self.Q = 0 # Accumulated time constraint violation deficit, corresponding to Q in the old model

    # Reward calculation method in the new model, corresponding to fun_C in BR__.py
    # strategy_list is a numpy array of shape [2, num_agents],
    # where strategy_list[0] is the d strategy for all agents, and strategy_list[1] is the f strategy for all agents.
    def calculate_reward(self, all_agents_strategy_list, M_cooperation, V): # Receives V parameter
        # Ensure all_agents_strategy_list is a numpy array
        all_agents_strategy_list = np.array(all_agents_strategy_list)

        d = all_agents_strategy_list[0].copy()
        f = all_agents_strategy_list[1].copy()
        
        # Get the index of the current agent
        i = self.agent_id

        d_local = d.copy()
        d_local[i] = 0 # d_local used to calculate sum_base in the original fun_C

        # Ensure S, G, Z, X, W, gama, omega, D_T are imported and initialized from environment
        # These are global variables and can be used directly

        sum_profit = Z[i][0] * fun_p(d, S, G)
        sum_base = X[0][i] * fun_p(d_local, S, G)
        
        sum_energy = f[i] * f[i] * d[i] * W[0][i]
        
        sum_time = omega * (D_T[0][i] - (yeta[0][i] * d[i] * S[0][i]) / (f[i] * F[0][i]))
        self.current_sum_time = -sum_time # Store the value of sum_time

        # Ensure M_cooperation is the correct cooperation matrix
        sum_cooperation = gama * (d[i] * S[0][i] * np.sum(M_cooperation[i]) - np.sum(d * S * M_cooperation[i]) + \
                                  f[i] * F[0][i] * np.sum(M_cooperation[i]) - np.sum(f * F * M_cooperation[i]))
        
        # Calculate the original reward component (main part of fun_C)
        original_reward_component = sum_profit + sum_base - sum_energy + sum_cooperation
        self.original_reward_component = original_reward_component # Store the original reward component
        
        # Calculate time constraint violation deficit
        if sum_time < 0:
            self.time_violation_deficit = abs(sum_time)
        else:
            self.time_violation_deficit = 0
        
        # According to the mathematical derivation from user feedback, the final reward is V * original_reward - actual_time * deficit_queue_length
        # Note: self.Q is used here because it is the accumulated deficit queue length.
        final_reward = original_reward_component + sum_time * self.Q / V
        
        self.sum_time = sum_time
        self.reward = final_reward # Update the agent's reward attribute
        
        return final_reward

    # Helper function for the optimizer to find the best strategy
    # df is an array containing [d, f]
    # all_agents_strategy_list is the current strategy list of all agents
    # agent_id is the index of the current agent
    # M_cooperation is the cooperation matrix
    def _fun_for_optimization(self, df, all_agents_strategy_list, agent_id, M_cooperation, V): # Receives V parameter
        temp_strategy_list = np.array(list(all_agents_strategy_list)) # Copy the strategy list
        temp_strategy_list[:, agent_id] = df # Update the current agent's strategy
        
        # Recalculate the original reward component and sum_time for fun_C
        d = temp_strategy_list[0].copy()
        f = temp_strategy_list[1].copy()
        
        i = agent_id

        d_local = d.copy()
        d_local[i] = 0
        
        sum_profit = Z[i][0] * fun_p(d, S, G)
        sum_base = X[0][i] * fun_p(d_local, S, G)
        
        sum_energy = f[i] * f[i] * d[i] * W[0][i]
        
        sum_time = omega * (D_T[0][i] - (yeta[0][i] * d[i] * S[0][i]) / (f[i] * F[0][i]))
        
        sum_cooperation = gama * (d[i] * S[0][i] * np.sum(M_cooperation[i]) - np.sum(d * S * M_cooperation[i]) + \
                                  f[i] * F[0][i] * np.sum(M_cooperation[i]) - np.sum(f * F * M_cooperation[i]))
        
        original_reward_component = sum_profit + sum_base - sum_energy + sum_cooperation
        
        # The optimization objective is to maximize (V * original_reward_component - sum_time * self.Q)
        # Note: self.Q here should be the Q value of the current agent before optimization, not after optimization.
        # Because Q is updated at the end of the time slot, the optimizer should use the current Q when searching for the best strategy within the current time slot.
        # For simplicity, the Agent instance's Q attribute is used directly here.
        # If more precision is needed, Q can be passed as a parameter.
        current_Q_for_optimization = self.Q # Use the current agent's Q value
        
        objective_value = original_reward_component + sum_time * current_Q_for_optimization / V
        
        return -objective_value # Return negative value because dual_annealing seeks to minimize

    # Finds and updates the best strategy for the current agent
    # all_agents_strategy_list is the current strategy list of all agents
    # M_cooperation is the cooperation matrix
    def refresh_best_strategy(self, all_agents_strategy_list, M_cooperation, V): # Receives V parameter
        lb = [1e-3] * 2 # Lower bound for d and f
        ub = [1] * 2   # Upper bound for d and f
        bnds = tuple([(lb[j], ub[j]) for j in range(len(lb))])

        # Use dual_annealing to find the strategy that maximizes reward
        res = opt.dual_annealing(self._fun_for_optimization, bnds, 
                                 args=(all_agents_strategy_list, self.agent_id, M_cooperation, V), # Pass V parameter
                                 maxiter=100) # Reduce iterations to speed up simulation, can be adjusted as needed
        
        new_d, new_f = res.x
        max_reward_objective = -res.fun # The optimizer returns the negative of the minimum, so take the negative to get the maximum reward

        # Update the current agent's strategy
        self.d = new_d
        self.f = new_f
        # self.reward should be updated to the final reward based on the new strategy and V.
        # Recalculate reward to ensure reward, time_violation_deficit, and current_sum_time are up-to-date.
        # Note: The all_agents_strategy_list passed here should be the list containing the new strategy.
        # To avoid circular dependencies, max_reward_objective is used directly as reward here.
        # Alternatively, calculate each agent's final reward uniformly in main.py after BR optimization.
        # Given the user's requirement for calculate_reward to update self.reward, the optimizer's result is used directly here.
        self.reward = max_reward_objective 
        return new_d, new_f, max_reward_objective

    # Update virtual queue Q
    def refresh_Q(self):
        # Q corresponds to the time constraint violation deficit; when there is a surplus, the queue shortens but does not go below 0.
        # current_sum_time > 0 indicates a surplus, < 0 indicates a deficit.
        self.Q = max(0, self.Q + self.current_sum_time)
