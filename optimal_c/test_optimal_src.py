from typing import List
import copy
import numpy as np
from scipy.stats import norm, laplace
from collections import defaultdict
from tqdm.contrib.concurrent import process_map


# optimal one step controller
ACCELS = np.linspace(-5, 5, 1024)

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
    turnCost = computeProbCosts(sim, probs)
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
        rec_move, rec_cost, ls = rec_controller.solve(remBudget/val)
        remBudget -= rec_cost * val
        if remBudget <= 0:
            return float('inf')
        for _ in range(val):
          costs.append(rec_cost)
    return turnCost + np.mean(costs)

def solve(sim, numRecSteps, nextController):
      lastAction = sim.action_history[-1]
      ls = []
      for action in [-0.05, -0.005, 0, 0.005, 0.05]:
        cost = evalOption(sim, lastAction + action, nextController, numRecSteps); ls.append((lastAction+action, cost))      
      
      action, minCost = min(ls, key=lambda x: x[1])
      while True:
        # fit a parabola
        actions, costs = zip(*ls)
        p = np.polyfit(actions, costs, 2)
        # find minimum
        minimum = -p[1]/(2*p[0])
        # if wrong shape then ret immediately
        if p[0]<0 or abs(minimum-lastAction)>0.05:
          break
        cost = evalOption(sim, minimum, nextController, numRecSteps); ls.append((minimum, cost))

        if cost < minCost:
          action = minimum
          minCost = cost
        if cost >= minCost - 0.01:
          break
      
      return action, minCost, ls


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
    self.ls = []
    bestOption = None
    bestCost = float('inf')
    lastMove = self.sim.action_history[-1]
    for option in linspace_laplace_halfstep(lastMove, 0.015, 20):
      probs = self.sim.getProbDistMid(option)
      cost = computeProbCosts(self.sim, probs)
      self.ls.append((option, cost))
      if cost < bestCost:
        bestCost = cost
        bestOption = option

    return bestOption, bestCost, []

class Controller1StepSingle:
  def __init__(self):
     self.IDX = 20
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    if self.IDX<100:
      ret = 0
    else:
      self.data = self.solve()
      ret = self.data[0]
    self.IDX+=1
    return ret
  def giveSim(self, sim_):
    self.sim = sim_

  def solve(self, budget=float('inf')):
    if budget <= 0:
      return None, float('inf')
    lastMove = self.sim.action_history[-1]
    probs = self.sim.getProbDistMid(lastMove)
    cost = computeProbCosts(self.sim, probs)
    return lastMove, cost, [(lastMove, cost)]

class Controller2StepFast:
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

  def solve(self, maxCost=float('inf')):
    return solve(self.sim, 4, Controller1StepSingle)

class Controller3StepFast:
  def __init__(self):
     self.IDX = 20
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    if self.IDX<100:
      ret = 0
    else:
      self.data = self.solve()
      ans, minCena, ls = self.data
      print(f"{ans=:.3f} {minCena=:.3f}")
      ret = self.data[0]
    self.IDX+=1
    return ret
  def giveSim(self, sim_):
    self.sim = sim_

  def solve(self, maxCost=float('inf')):
    return solve(self.sim, 4, Controller2StepFast)

# Standalone function to allow pickling
def evaluate_option(option, sim, max_cost=float('inf')):
    probs = sim.getProbDistMid(option)
    turnCost =  computeProbCosts(sim, probs)
    max_cost -= turnCost
    if max_cost <= 0:
        return None, float('inf')

    costs = []
    counts = defaultdict(int)
    for idx in integral_sample(probs, 4):
        counts[idx] += 1

    rem_budget = 4 * max_cost
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

        c2s = Controller3StepFast()
        c2s.giveSim(sim2)
        _, c2, _ = c2s.solve(rem_budget/val)
        rem_budget -= c2 * val
        if rem_budget <= 0:
            return None, float('inf')
        for _ in range(val):
            costs.append(c2)

    avg_cost = np.mean(costs)
    # print(f"Option: {option:.3f}, AvgCost: {(turnCost + avg_cost)/3:.2f}")
    return option, turnCost + avg_cost

class Controller4StepParallel:
    def __init__(self):
        self.IDX = 20

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        if self.IDX < 100:
            ret = 0
        else:
            self.data = self.solve()
            ret = self.data[0]
        self.IDX += 1
        return ret

    def giveSim(self, sim_):
        self.sim = sim_

    def solve(self):
        lastAction = self.sim.action_history[-1]
        options = linspace_laplace_halfstep(lastAction, 0.015, 16)
        
        results = process_map(evaluate_option, options, [self.sim] * len(options), max_workers=16, chunksize=1)

        bestOption, bestCost = min(results, key=lambda x: x[1])
        self.ls = results
        print(f"Best option: {bestOption:.3f}, Best cost: {bestCost:.2f}")

        return bestOption, bestCost, []