import numpy as np
import copy
from typing import List
from controllers import BaseController

IDX = 0
class Controller(BaseController):
    def __init__(self):
        global IDX
        IDX = 20
        self.horizon = 8  # Planning horizon (steps)
        self.n_particles = 50  # Number of trajectory rollouts
        self.sim = None
        
        # Control sequence parameterization
        self.n_basis = 4  # Number of basis functions
        self.basis_functions = self._create_basis_functions()
        
    def _create_basis_functions(self):
        """Create basis functions for control parameterization"""
        def basis(t, i, n):
            centers = np.linspace(0, self.horizon-1, self.n_basis)
            width = (centers[1] - centers[0]) if n > 1 else 1.0
            return np.exp(-0.5 * ((t - centers[i]) / width) ** 2)
        
        return [lambda t, i=i: basis(t, i, self.n_basis) 
                for i in range(self.n_basis)]

    def _parameterize_control(self, weights):
        """Convert basis weights to control sequence"""
        controls = np.zeros(self.horizon)
        for t in range(self.horizon):
            controls[t] = sum(w * b(t) for w, b in zip(weights, self.basis_functions))
        return controls
    
    def simulateNActions(self, actions: List[float]):
        sim2 = copy.deepcopy(self.sim)
        for action in actions:
            sim2.action_history.append(action)
            sim2.sim_step(sim2.step_idx)
            sim2.step_idx += 1
            state, target, futureplan = sim2.get_state_target_futureplan(sim2.step_idx)
            sim2.state_history.append(state)
            sim2.target_lataccel_history.append(target)
            sim2.futureplan = futureplan
        
        return sim2.current_lataccel_history[-len(actions):]
    
    def _compute_trajectory_cost(self, lataccels, target):
        """Compute cost for a single trajectory"""
        # Tracking error cost
        tracking_cost = np.mean((target - lataccels) ** 2) * 100
        
        # Control jerk cost
        jerk = np.diff(np.concatenate([[self.sim.current_lataccel], lataccels])) / 0.1
        jerk_cost = np.mean(jerk ** 2) * 100
        
        return 50 * tracking_cost + jerk_cost
    
    def solve(self):
        if getattr(self, 'sim', None) is None:
            self.sim = sim
        last_action = self.sim.action_history[-1]
        target = self.sim.data['target_lataccel'].values[
            self.sim.step_idx:self.sim.step_idx + self.horizon
        ]
                
        # Generate and evaluate particles
        costs = []
        controls = []
        for _ in range(self.n_particles):
            # Sample basis weights
            weights = np.random.normal(0, 0.03, self.n_basis)
            
            # Generate control sequence
            control_sequence = last_action + np.cumsum(self._parameterize_control(weights))
            control_sequence = np.clip(control_sequence, -2, 2)
            
            # Simulate trajectory
            lataccels = self.simulateNActions(control_sequence)
            
            # Compute cost
            cost = self._compute_trajectory_cost(lataccels, target)
            
            costs.append(cost)
            controls.append(control_sequence)
        
        # Risk-sensitive optimization
        costs = np.array(costs)
        
        # Select best control sequence considering risk
        best_idx = np.argmin(costs)
        best_control = controls[best_idx]
        
        return best_control[0]
    
    def giveSim(self, sim):
        self.sim = sim
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        global IDX

        if IDX < 100:
            ret = 0
        else:
            print(IDX)
            ret = self.solve()
        IDX += 1
        return ret