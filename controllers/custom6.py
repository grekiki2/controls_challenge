from controllers import BaseController
from typing import List
import copy
import itertools
import numpy as np

class PIDController(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,):
    self.p = 0.3
    self.i = 0.05
    self.d = -0.1
    self.error_integral = 0
    self.prev_error = 0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
      error = (target_lataccel - current_lataccel)
      self.error_integral += error
      error_diff = error - self.prev_error
      self.prev_error = error
      return self.p * error + self.i * self.error_integral + self.d * error_diff


IDX = 0
DEL_T = 0.1
sim = None

def simulateNPidAfterSteps(actions:list[float], n:int):
  sim2 = copy.deepcopy(sim)
  pid2 = copy.deepcopy(pid)
  pidAct = 0
  for action in actions:
    sim2.action_history.append(action)
    sim2.sim_step(sim2.step_idx)
    sim2.step_idx += 1
    
    state, target, futureplan = sim2.get_state_target_futureplan(sim2.step_idx)
    sim2.state_history.append(state)
    sim2.target_lataccel_history.append(target)
    sim2.futureplan = futureplan
    pidAct = pid2.update(sim2.target_lataccel_history[sim2.step_idx], sim2.current_lataccel, sim2.state_history[sim2.step_idx], sim2.futureplan)

  for _ in range(n):
    sim2.action_history.append(pidAct)
    sim2.sim_step(sim2.step_idx)
    sim2.step_idx += 1

    state, target, futureplan = sim2.get_state_target_futureplan(sim2.step_idx)
    sim2.state_history.append(state)
    sim2.target_lataccel_history.append(target)
    sim2.futureplan = futureplan
    pidAct = pid2.update(sim2.target_lataccel_history[sim2.step_idx], sim2.current_lataccel, sim2.state_history[sim2.step_idx], sim2.futureplan)
  
  return sim2.current_lataccel_history[-n-len(actions):]


pid = PIDController()
pid_ret = 0
NEXT_STEPS = 10
HORIZON = 5

class Controller(BaseController):
  def __init__(self):
    global IDX
    IDX=20
    self.n_samples = 100
    self.n_elite = self.n_samples//10
    self.initial_std = 0.02
    self.mean = np.zeros(HORIZON)
    self.std = self.initial_std * np.ones(HORIZON)

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    global IDX
    pid.update(target_lataccel, current_lataccel, state, future_plan)

    if IDX<100:
      ret = 0
    else:
      print(IDX)
      ret = self.solve()
    IDX+=1
    return ret
  def giveSim(self, sim_):
    global sim
    sim = sim_

  def solve(self):
    last_action = sim.action_history[-1]
    last_lataccel = sim.current_lataccel_history[-1]
    target = sim.data['target_lataccel'].values[sim.step_idx: sim.step_idx + HORIZON+NEXT_STEPS]
    self.mean = np.roll(self.mean, -1)
    self.mean[-1] = 0
    self.std = self.initial_std * np.ones(HORIZON)

    iters = 10 if IDX == 100 else 1
    for _ in range(iters):
        # Sample action sequences
        samples = np.random.normal(self.mean, self.std, (self.n_samples, HORIZON))
        costs = np.zeros(self.n_samples)

        # Evaluate each sequence
        for i in range(self.n_samples):
            actions = [last_action + np.sum(samples[i,:j+1]) for j in range(HORIZON)]
            lataccels = simulateNPidAfterSteps(actions, NEXT_STEPS)
            
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
        print(f"Mean: {self.mean*1000}")
        print(f"Std: {self.std*1000}")
        print(f"Elite costs {np.mean(costs[elite_idx])}")    

    # Return first action from mean trajectory
    return last_action + self.mean[0]