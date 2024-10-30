from controllers import BaseController
from typing import List
import copy
import itertools
import numpy as np

IDX = 0
DEL_T = 0.1
sim = None
def simulateNActions(actions:List[float]):
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
    IDX=20

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    global IDX

    if IDX<100:
      ret = 0
    else:
      print(IDX)
      ret = solve()
    IDX+=1
    return ret
  def giveSim(self, sim_):
    global sim
    sim = sim_


STEPS = [3, 7]
ITERS = 50
OPTS = [
  [-0.1, -0.05, 0, 0.05, 0.1],
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
      perm_cost = np.mean([50*c[0]+c[1] for c in perm_costs[perm]])
      ucb = perm_cost - 0.8*np.sqrt(np.log(n))/len(perm_costs[perm])*perm_cost
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
    cost = np.mean([50*c[0]+c[1] for c in perm_costs[perm]])
    # print(f"{cost:.2f}", len(perm_costs[perm]), perm)
    if cost < bestCost:
      bestCost = cost
      bestPerm = perm
  # print(f"Best perm: {([f'{b:.3f}' for b in bestPerm])}")
  if bestPerm[0] == 0:
    corr = 1/1.2 if IDX>3 else 0.5
  elif bestPerm[0] in (OPTS[0][0], OPTS[0][-1]): # we want more movement
    corr = 1.2 if IDX>3 else 2
  else:
    corr = 1
  for i in range(len(OPTS)):
    for j in range(len(OPTS[i])):
      OPTS[i][j] *= corr

  # print(last_action+bestPerm[0])
  return last_action + bestPerm[0]

controller = Controller()