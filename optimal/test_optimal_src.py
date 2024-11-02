from typing import List
import copy
import numpy as np
from scipy.stats import norm, laplace
from collections import defaultdict
from tqdm.contrib.concurrent import process_map


# optimal one step controller
ACCELS = np.linspace(-5, 5, 1024)

def linspace_normal_halfstep(mean, std, num):
    # Generate values centered in the distribution by using half-step offset
    uniform_points = np.linspace(0, 1, num + 1)[1:] - 0.5 / num
    # Apply the inverse CDF (ppf) to get normally distributed values
    normal_points = norm.ppf(uniform_points, loc=mean, scale=std)
    return normal_points

def linspace_laplace_halfstep(mean, scale, num):
    # Generate uniformly spaced points between 0 and 1, shifted by half a step
    uniform_points = np.linspace(0, 1, num + 1)[1:] - 0.5 / num
    # Apply the inverse CDF (ppf) to get Laplace-distributed values 
    laplace_points = laplace.ppf(uniform_points, loc=mean, scale=scale)
    return laplace_points

def computeProbCosts(sim, probs):
    prev_accel = sim.current_lataccel_history[-1]
    target = sim.target_lataccel_history[-1]
    
    angle_costs = 100 * (ACCELS - target) ** 2    
    jerk_costs = 100 * ((ACCELS - prev_accel) / 0.1) ** 2    
    total_costs = probs * (50 * angle_costs + jerk_costs)
    
    return np.sum(total_costs)

def computeProbCosts2(sim, probs):
    prev_accel = sim.current_lataccel_history[-1]
    target = sim.target_lataccel_history[-1]
    cost = 0
    for i, prob in enumerate(probs):
        lataccel = ACCELS[i]
        angle_cost = 100 * (lataccel - target) ** 2
        jerk_cost = 100 * ((lataccel - prev_accel) / 0.1) ** 2
        cost += prob * (50 * angle_cost + jerk_cost)
    return cost

def integral_sample(probs, num_samples):
    cdf = np.cumsum(probs)
    cdf /= cdf[-1]  # Normalize CDF to ensure it reaches 1
    # Generate evenly spaced values between 0 and 1 (half-step offset)
    uniform_samples = (np.arange(num_samples) + 0.5) / num_samples
    # Find indices in the CDF that correspond to the uniformly spaced samples
    indices = np.searchsorted(cdf, uniform_samples)
    return indices

def evalOption(sim, option, recController, distSteps, max_cost=float('inf')):
    probs = sim.getProbDistMid(option)
    # Za budget bi lahko pri vsakem probs samplu kalkulirali cost, kar pa je manj natančno.
    # Zato cost poračunamo direktno iz distribucije.
    turnCost =  computeProbCosts(sim, probs)
    max_cost -= turnCost
    if max_cost <= 0:
        return float('inf')

    costs = []
    counts = defaultdict(int)
    for idx in integral_sample(probs, distSteps):
        counts[idx] += 1
    remBudget = distSteps * max_cost
    for idx in counts.keys():
        val = counts[idx]
        # tale branch lahko porabi vec od budgeta ker je vazna povprecna cena
        # Dobi pa omejitev budget*distSteps/val.
        nextAccel = ACCELS[idx]

        sim2 = copy.deepcopy(sim)
        sim2.action_history.append(option)
        sim2.current_lataccel_history.append(nextAccel)
        sim2.step_idx += 1

        state, target, futureplan = sim2.get_state_target_futureplan(sim2.step_idx)
        sim2.state_history.append(state)
        sim2.target_lataccel_history.append(target)
        sim2.futurplan = futureplan
        rec_controller = recController()
        rec_controller.giveSim(sim2)
        # solver dobi omejen budget. Je le en izmed branchov distribucije
        # zato ima potencialno budget kar velik
        rec_move, rec_cost = rec_controller.solve(remBudget/val)
        remBudget -= rec_cost * val
        if remBudget <= 0:
            return float('inf')
        for _ in range(val):
          costs.append(rec_cost)
    return turnCost + np.mean(costs)

class Controller1Step:
  def __init__(self):
     self.IDX = 20
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    if self.IDX<100:
      ret = 0
    else:
      self.data = self.solve()
    self.IDX+=1
    return 2
  def giveSim(self, sim_):
    self.sim = sim_

  # single step controller cannot really help itself with budget
  def solve(self, budget=float('inf')):
    if budget <= 0:
      return None, float('inf')
    self.logs = {}
    ls = []
    bestOption = None
    bestCost = float('inf')
    lastMove = self.sim.action_history[-1]
    for option in np.hstack([np.linspace(-2, 2, 100), linspace_laplace_halfstep(lastMove, 0.0123, 20)]):
      probs = self.sim.getProbDistMid(option)
      cost = computeProbCosts(self.sim, probs)
      ls.append((option, cost))
      if cost < bestCost:
        bestCost = cost
        bestOption = option

    self.logs['cene'] = ls
    return bestOption, bestCost
  
class Controller1StepOptimized:
  def __init__(self):
     self.IDX = 20
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    if self.IDX<100:
      ret = 0
    else:
      self.data = self.solve()
    self.IDX+=1
    return 2
  def giveSim(self, sim_):
    self.sim = sim_

  # single step controller cannot really help itself with budget
  def solve(self, budget=float('inf')):
    if budget <= 0:
      return None, float('inf')
    self.logs = {}
    ls = []
    lastAction = self.sim.action_history[-1]
    bestOption = None
    bestCost = float('inf')
    for option in np.hstack([np.linspace(-2, 2, 6), linspace_laplace_halfstep(lastAction, 0.015, 3)]):
      probs = self.sim.getProbDistMid(option)
      cost = computeProbCosts(self.sim, probs)
      ls.append((option, cost))
      if cost < bestCost:
        bestCost = cost
        bestOption = option
    
    # if the best option is close to the last move, return it since we explored 
    # close options already
    if abs(bestOption - lastAction) < 0.05:
      self.logs['cene'] = ls
      return bestOption, bestCost
    
    for option in np.linspace(bestOption-0.4, bestOption+0.4, 5):
      if option == bestOption:
        continue
      probs = self.sim.getProbDistMid(option)
      cost = computeProbCosts(self.sim, probs)
      ls.append((option, cost))
      if cost < bestCost:
        bestCost = cost
        bestOption = option

    self.logs['cene'] = ls
    return bestOption, bestCost
  
class Controller2Step:
  def __init__(self):
     self.IDX = 20
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    if self.IDX<100:
      ret = 0
    else:
      self.data = self.solve()
    self.IDX+=1
    return 2
  def giveSim(self, sim_):
    self.sim = sim_

  def solve(self):
    lastAction = self.sim.action_history[-1]
    bestOption = None
    bestCost = float('inf')
    self.ls = []
    options = np.hstack([np.linspace(-2, 2, 41), linspace_laplace_halfstep(lastAction, 0.015, 10)])
    for option in options:
      cost = evalOption(self.sim, option, Controller1StepOptimized, 10000)
      print(f"Option: {option:.2f}, Cost: {cost:.2f}")
      self.ls.append((option, cost))
      if cost < bestCost:
        bestCost = cost
        bestOption = option
    return bestOption, bestCost
  
class Controller2StepOptimized:
  def __init__(self):
     self.IDX = 20
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    if self.IDX<100:
      ret = 0
    else:
      self.data = self.solve()
    self.IDX+=1
    return 2
  def giveSim(self, sim_):
    self.sim = sim_
  def solve(self):
    lastAction = self.sim.action_history[-1]
    bestOption = None
    bestCost = float('inf')
    self.ls = []
    for option in np.hstack([np.linspace(-2, 2, 6), linspace_laplace_halfstep(lastAction, 0.015, 5)]):
      cost = evalOption(self.sim, option, Controller1StepOptimized, 10)
      # print(f"Option: {option:.2f}, Cost: {cost:.2f}")
      self.ls.append((option, cost))
      if cost < bestCost:
        bestCost = cost
        bestOption = option
    if abs(bestOption - lastAction) < 0.05:
      return bestOption, bestCost
    
    for option in np.linspace(bestOption-0.4, bestOption+0.4, 5):
      if option == bestOption:
        continue
      cost = evalOption(self.sim, option, Controller1StepOptimized, 10)
      self.ls.append((option, cost))
      if cost < bestCost:
        bestCost = cost
        bestOption = option
    return bestOption, bestCost

class Controller2StepBudget:
  def __init__(self):
     self.IDX = 20
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    if self.IDX<100:
      ret = 0
    else:
      self.data = self.solve()
    self.IDX+=1
    return 2
  def giveSim(self, sim_):
    self.sim = sim_

  def solve(self, max_allowed_cost=float('inf')):
    lastAction = self.sim.action_history[-1]
    bestOption = None
    bestCost = max_allowed_cost
    self.ls = []
    actions = np.hstack([np.linspace(-2, 2, 6), linspace_laplace_halfstep(lastAction, 0.015, 5)]).tolist()
    actions.sort(key=lambda x: abs(x - lastAction))
    for action in actions:
      cost = evalOption(self.sim, action, Controller1StepOptimized, 10, bestCost)
      # print(f"Action: {action:.2f}, AvgCost: {cost/2:.2f}")
      self.ls.append((action, cost))
      if cost < bestCost:
        bestCost = cost
        bestOption = action
    if bestOption is None or abs(bestOption - lastAction) < 0.05:
      return bestOption, bestCost
    
    for action in np.linspace(bestOption-0.4, bestOption+0.4, 5):
      if action == bestOption:
        continue
      cost = evalOption(self.sim, action, Controller1StepOptimized, 10, bestCost)
      # print(f"Action: {action:.2f}, AvgCost: {cost/2:.2f}")
      self.ls.append((action, cost))
      if cost < bestCost:
        bestCost = cost
        bestOption = action
    return bestOption, bestCost

# 3 step needs custom eval even though it's the same for pickling purposes
def evaluate_option(option, sim, max_cost):
    probs = sim.getProbDistMid(option)
    turnCost =  computeProbCosts(sim, probs)
    max_cost -= turnCost
    if max_cost <= 0:
        return None, float('inf')

    costs = []
    counts = defaultdict(int)
    for idx in integral_sample(probs, 10):
        counts[idx] += 1

    rem_budget = 10 * max_cost
    for idx in counts.keys():
        val = counts[idx]
        nextAccel = ACCELS[idx]

        sim2 = copy.deepcopy(sim)
        sim2.action_history.append(option)
        sim2.current_lataccel_history.append(nextAccel)
        sim2.step_idx += 1

        state, target, futureplan = sim2.get_state_target_futureplan(sim2.step_idx)
        sim2.state_history.append(state)
        sim2.target_lataccel_history.append(target)
        sim2.futurplan = futureplan

        # c2s = Controller2StepOptimized()
        c2s = Controller2StepBudget()
        c2s.giveSim(sim2)
        _, c2 = c2s.solve(rem_budget/val)
        rem_budget -= c2 * val
        if rem_budget <= 0:
            return None, float('inf')
        for _ in range(val):
            costs.append(c2)

    avg_cost = np.mean(costs)
    # print(f"Option: {option:.3f}, AvgCost: {(turnCost + avg_cost)/3:.2f}")
    return option, turnCost + avg_cost

class Controller3StepParallel:
    def __init__(self):
        self.IDX = 20

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        if self.IDX < 100:
            ret = 0
        else:
            self.data = self.solve()
        self.IDX += 1
        return 2

    def giveSim(self, sim_):
        self.sim = sim_

    def solve(self, max_allowed_cost=float('inf')):
        lastAction = self.sim.action_history[-1]
        options = np.hstack([np.linspace(-2, 2, 11), linspace_laplace_halfstep(lastAction, 0.015, 5)])
        
        results = process_map(evaluate_option, options, [self.sim] * len(options), [max_allowed_cost] * len(options), max_workers=16, chunksize=1)

        bestOption, bestCost = min(results, key=lambda x: x[1])
        self.ls = results

        return bestOption, bestCost

class Controller3StepBudget:
  def __init__(self):
     self.IDX = 20
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    if self.IDX<100:
      ret = 0
    else:
      self.data = self.solve()
    self.IDX+=1
    return 2
  def giveSim(self, sim_):
    self.sim = sim_

  def solve(self, max_allowed_cost=float('inf')):
    lastAction = self.sim.action_history[-1]
    bestOption = None
    bestCost = max_allowed_cost
    self.ls = []
    actions = np.hstack([np.linspace(-2, 2, 6), linspace_laplace_halfstep(lastAction, 0.015, 5)]).tolist()
    actions.sort(key=lambda x: abs(x - lastAction))
    for action in actions:
      cost = evalOption(self.sim, action, Controller2StepBudget, 10, bestCost)
      print(f"Action: {action:.3f}, AvgCost: {cost/3:.2f}")
      self.ls.append((action, cost))
      if cost < bestCost:
        bestCost = cost
        bestOption = action
    
    if bestOption is None or abs(bestOption - lastAction) < 0.05:
      return bestOption, bestCost

    for action in np.linspace(bestOption-0.4, bestOption+0.4, 5):
      if action == bestOption:
        continue
      cost = evalOption(self.sim, action, Controller2StepBudget, 10, bestCost)
      print(f"Action: {action:.3f}, AvgCost: {cost/3:.2f}")
      self.ls.append((action, cost))
      if cost < bestCost:
        bestCost = cost
        bestOption = action
    return bestOption, bestCost


class Controller4Step:
  def __init__(self):
     self.IDX = 20
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    if self.IDX<100:
      ret = 0
    else:
      self.data = self.solve()
    self.IDX+=1
    return 2
  def giveSim(self, sim_):
    self.sim = sim_

  def solve(self):
    lastAction = self.sim.action_history[-1]
    bestOption = None
    bestCost = float('inf')
    self.ls = []
    actions = np.hstack([np.linspace(-2, 2, 6), linspace_laplace_halfstep(lastAction, 0.015, 5)]).tolist()
    actions.sort(key=lambda x: abs(x - lastAction))
    for action in actions:
      cost = evalOption(self.sim, action, Controller3StepParallel, 10, bestCost)
      print(f"Action v4!: {action:.3f}, AvgCost: {cost/4:.2f}")
      self.ls.append((action, cost))
      if cost < bestCost:
        bestCost = cost
        bestOption = action
    
    if bestOption is None or abs(bestOption - lastAction) < 0.05:
      return bestOption, bestCost

    for action in np.linspace(bestOption-0.4, bestOption+0.4, 5):
      if action == bestOption:
        continue
      cost = evalOption(self.sim, action, Controller3StepParallel, 10, bestCost)
      print(f"Action: {action:.3f}, AvgCost: {cost/4:.2f}")
      self.ls.append((action, cost))
      if cost < bestCost:
        bestCost = cost
        bestOption = action
    return bestOption, bestCost
