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

def generateActions(init, delta):
  actions = []
  for _ in range(3):
    init += delta
    actions.append(init)
  for _ in range(7):
    actions.append(init)
  return actions

ITERS = 30
def solve():
  global coords
  last_action = sim.action_history[-1]
  last_lataccel = sim.current_lataccel_history[-1]
  target = sim.data['target_lataccel'].values[sim.step_idx: sim.step_idx + 3+7]

  best = None
  best_cost = float('inf')
  for _ in range(ITERS):
    rand = np.random.normal(0, 0.01, 1)[0]
    actions = generateActions(last_action, rand)
    lataccels = simulateNActions(actions)

    angle_cost = 100 * np.mean((target - lataccels)**2)
    jerk_cost = 100 * np.mean((np.diff([last_lataccel] + lataccels) / DEL_T)**2)

    if 50*angle_cost + jerk_cost < best_cost:
      best = rand
      best_cost = 50*angle_cost + jerk_cost

  return last_action + best
