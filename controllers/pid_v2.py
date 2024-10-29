from collections import namedtuple
from . import BaseController

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['target', 'roll_lataccel', 'v_ego', 'a_ego'])

class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,):
    self.p = 0.3
    self.i = 0.05
    self.d = -0.1
    self.error_integral = 0
    self.prev_error = 0

    self.idx = 20

  def update(self, target_lataccel:float, current_lataccel:float, state:State, future_plan:FuturePlan):
    error = target_lataccel - current_lataccel

    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error
    
    control = self.p * error + self.i * self.error_integral + self.d * error_diff
    self.idx += 1
    return control

