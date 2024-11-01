from . import BaseController
import numpy as np

class Controller(BaseController):
  def __init__(self,):
    self.p = 0.15
    self.d = 0.15
    self.prev_error = 0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    target_lataccel = future_plan[0][1]
    error = target_lataccel - current_lataccel
    error_diff = error - self.prev_error
    self.prev_error = error
    return sim.action_history[-1] + self.p * error + self.d * error_diff
    
  def giveSim(self, sim_):
    global sim
    sim = sim_