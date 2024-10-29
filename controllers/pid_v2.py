from . import BaseController

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

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    # if self.idx == 20:
    #   print("target_lataccel: ", target_lataccel)
    #   print("current_lataccel: ", current_lataccel)
    #   print("state: ", state)
    #   print("future_plan: ", future_plan)
    # if self.idx == 100:
    #   print(current_lataccel)
    #   print(target_lataccel)
    error = target_lataccel - current_lataccel

    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error
    
    control = self.p * error + self.i * self.error_integral + self.d * error_diff
    self.idx += 1
    return control

      
