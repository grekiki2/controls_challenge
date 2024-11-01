from controllers import BaseController
from typing import List
import copy
import numpy as np
np.set_printoptions(precision=4)

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
        self.horizon = 15
        self.n_samples = 200
        self.n_elite = self.n_samples//10
        self.initial_std = 0.02
        self.mean = np.zeros(self.horizon)
        self.std = self.initial_std * np.ones(self.horizon)

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        global IDX

        if IDX < 100:
            ret = 0
        else:
            print()
            print(IDX)
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

        # rollback one step
        self.mean = np.roll(self.mean, -1)
        self.mean[-1] = 0
        self.std = np.roll(self.std, -1)
        self.std[-1] = self.initial_std
        self.std = np.maximum(self.std, self.initial_std/5)

        iters = 10 if IDX == 100 else 3
        for _ in range(iters):
            # Sample action sequences
            samples = np.random.normal(self.mean, self.std, (self.n_samples, self.horizon))
            costs = np.zeros(self.n_samples)

            # Evaluate each sequence
            for i in range(self.n_samples):
                actions = [last_action + np.sum(samples[i,:j+1]) for j in range(self.horizon)]
                lataccels = simulateNActions(actions)
                
                # Calculate costs
                angle_cost = 100 * np.mean((target - lataccels)**2)
                jerk_cost = 100 * np.mean((np.diff([last_lataccel] + lataccels) / DEL_T)**2)
                costs[i] = 50 * angle_cost + jerk_cost

            # Get elite samples
            elite_idx = np.argsort(costs)[:self.n_elite]
            elite_samples = samples[elite_idx]

            # Update distribution parameters
            self.mean = np.mean(elite_samples, axis=0)
            self.std = np.std(elite_samples, axis=0)
            print(f"Mean: {self.mean[:5]*1000}")
            print(f"Std: {self.std[:5]*1000}")
            print(f"Elite costs {np.mean(costs[elite_idx])}")

        # predict value
        actions = [last_action + np.sum(self.mean[:j+1]) for j in range(self.horizon)]
        costs = np.zeros(100)
        for i in range(100):
            lataccels = simulateNActions(actions)
            angle_cost = 100 * ((target - lataccels)**2)[0]
            jerk_cost = 100 * ((np.diff([last_lataccel] + lataccels) / DEL_T)**2)[0]
            costs[i] = 50 * angle_cost + jerk_cost
        print()

        print(f"Predicted next cost: {np.mean(costs)}")
        print(f"Predicted var: {np.std(costs)}")
        print(f"Predicted action: {actions[0]}")
        

        # Return first action from mean trajectory
        return last_action + self.mean[0]
    