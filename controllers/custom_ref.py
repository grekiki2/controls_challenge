from controllers import BaseController
from typing import List
import copy
import numpy as np

IDX = 0
DEL_T = 0.1
sim = None

def simulateNActions(actions: List[float]):
    sim2 = copy.deepcopy(sim)
    for action in actions:
        sim2.action_history.append(action)
        sim2.sim_step(sim2.step_idx)
        sim2.step_idx += 1
        state, target, futureplan = sim2.get_state_target_futureplan(sim2.step_idx)
        sim2.state_history.append(state)
        sim2.target_lataccel_history.append(target)
        sim2.futureplan = futureplan
    
    return sim2.current_lataccel_history[-len(actions):]

class Controller(BaseController):
    def __init__(self):
        global IDX
        IDX = 20
        self.horizon = 10  # Plan 10 steps ahead
        self.n_samples = 100  # Number of samples per iteration
        self.n_elite = 10  # Number of elite samples to keep
        self.n_iterations = 5  # Number of CEM iterations
        self.initial_std = 0.01  # Initial standard deviation for normal distribution

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        global IDX

        if IDX < 100:
            ret = 0
        else:
            ret = self.solve()
        IDX += 1
        return ret

    def giveSim(self, sim_):
        global sim
        sim = sim_

    def solve(self):
        last_action = sim.action_history[-1]
        last_lataccel = sim.current_lataccel_history[-1]
        target = sim.data['target_lataccel'].values[sim.step_idx: sim.step_idx + self.horizon]

        # Initialize distribution parameters
        mean = np.zeros(self.horizon)
        std = self.initial_std * np.ones(self.horizon)

        best_sequence = None
        best_cost = float('inf')

        for _ in range(self.n_iterations):
            # Sample action sequences
            samples = np.random.normal(mean, std, (self.n_samples, self.horizon))
            costs = np.zeros(self.n_samples)

            # Evaluate each sequence
            for i in range(self.n_samples):
                actions = [last_action + np.sum(samples[i,:j+1]) for j in range(self.horizon)]
                lataccels = simulateNActions(actions)
                
                # Calculate costs
                angle_cost = 100 * np.mean((target - lataccels)**2)
                jerk_cost = 100 * np.mean((np.diff([last_lataccel] + lataccels) / DEL_T)**2)
                costs[i] = 50 * angle_cost + jerk_cost

                # Track best sequence
                if costs[i] < best_cost:
                    best_cost = costs[i]
                    best_sequence = actions

            # Get elite samples
            elite_idx = np.argsort(costs)[:self.n_elite]
            elite_samples = samples[elite_idx]

            # Update distribution parameters
            mean = np.mean(elite_samples, axis=0)
            std = np.std(elite_samples, axis=0) + 1e-6  # Add small constant to prevent collapse

        # Return first action from best sequence
        return best_sequence[0]