from collections import namedtuple
from common import *
import itertools
import copy
sim = None
class BaseController:
  def update(self, target_lataccel:float, current_lataccel:float, state:State, active:bool, last_action:float) -> float:
    raise NotImplementedError

class OpenController(BaseController):
  def update(self, target_lataccel, current_lataccel, state, active, last_action):
    return target_lataccel

class SimpleController(BaseController):
  def update(self, target_lataccel, current_lataccel, state, active, last_action):
    return (target_lataccel - current_lataccel) * 0.3

class Controller(BaseController):
  def __init__(self):
    self.kp = 0.05
    self.ki = 0.1
    self.kd = 0.01
    self.prev_error = 0
    self.integral = 0

  def update(self, target_lataccel, current_lataccel, state, active, last_action):
    if not active:
      return
    error = target_lataccel - current_lataccel
    self.integral += error
    derivative = error - self.prev_error
    self.prev_error = error
    return self.kp * error + self.ki * self.integral + self.kd * derivative

DELTA = 10/1024
class Controller2(BaseController):
  def __init__(self):
    pass

  def update(self, target_lataccel, current_lataccel, state, active, last_action):
    return solve()

def simulateNActions(actions:List[float]):
  sim2 = copy.deepcopy(sim)
  for action in actions:
    sim2.action_history.append(action)
    sim2.sim_step(sim2.step_idx)
    sim2.step_idx += 1
  
  return sim2.current_lataccel_history[-len(actions):]

STEPS = [3, 3, 4]
ITERS = 30
OPTS = [
  [-0.02, 0, 0.02],
  [-0.02, 0, 0.02],
  [0.0],
]

def solve():
  last_action = sim.action_history[-1]
  last_lataccel = sim.current_lataccel_history[-1]
  target = sim.data['target_lataccel'].values[sim.step_idx: sim.step_idx + sum(STEPS)]

  perm_act = {}
  perms = list(itertools.product(*OPTS))
  for perm in perms:
    actions = []
    curAction = last_action
    for num, p in zip(STEPS, perm):
      for _ in range(num):
        curAction += p
        actions.append(curAction)

    perm_act[perm] = actions
  
  perm_costs = {perm:[] for perm in perms}
  for perm in perms:
    lataccels = simulateNActions(perm_act[perm])
    angle_cost = 100*np.mean((target - lataccels)**2)
    jerk_cost = 100*np.mean((np.diff([last_lataccel]+lataccels) / DEL_T)**2)
    perm_costs[perm].append((angle_cost, jerk_cost))
  
  for n in range(len(perms), ITERS+len(perms)):
    permToCheck = None
    bestCost = 1e9
    for perm in perms:
      perm_cost = np.mean([5*c[0]+c[1] for c in perm_costs[perm]])
      ucb = perm_cost - 1.1*np.sqrt(np.log(n))/len(perm_costs[perm])*perm_cost
      if ucb < bestCost:
        bestCost = ucb
        permToCheck = perm
    lataccels = simulateNActions(perm_act[permToCheck])
    angle_cost = 100*np.mean((target - lataccels)**2)
    jerk_cost = 100*np.mean((np.diff([last_lataccel]+lataccels) / DEL_T)**2)
    perm_costs[permToCheck].append((angle_cost, jerk_cost))


  bestPerm = None
  bestCost = 1e9
  for perm in perms:
    cost = np.mean([5*c[0]+c[1] for c in perm_costs[perm]])
    # print(perm, f"{cost:.2f}", len(perm_costs[perm]))
    if cost < bestCost:
      bestCost = cost
      bestPerm = perm
  print()
  print(f"Best perm: {([f'{b:.3f}' for b in bestPerm])}")
    
  return last_action + bestPerm[0]

CONTROLLERS = {
  'open': OpenController,
  'simple': SimpleController,
  'c1': Controller,
  'c2': Controller2,
}
