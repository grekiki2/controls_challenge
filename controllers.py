from collections import namedtuple
from common import *

class BaseController:
  def update(self, target_lataccel:float, current_lataccel:float, state:State, active:bool) -> float:
    raise NotImplementedError


class OpenController(BaseController):
  def update(self, target_lataccel, current_lataccel, state, active):
    return target_lataccel


class SimpleController(BaseController):
  def update(self, target_lataccel, current_lataccel, state, active):
    return (target_lataccel - current_lataccel) * 0.3

class Controller(BaseController):
  def __init__(self):
    self.kp = 0.05
    self.ki = 0.1
    self.kd = 0.01
    self.prev_error = 0
    self.integral = 0

  def update(self, target_lataccel, current_lataccel, state, active):
    if not active:
      return
    error = target_lataccel - current_lataccel
    self.integral += error
    derivative = error - self.prev_error
    self.prev_error = error
    return self.kp * error + self.ki * self.integral + self.kd * derivative

class Controller2(BaseController):
  def __init__(self):
    pass
  def update(self, target_lataccel, current_lataccel, state, active):
    roll_lataccel, v_ego, a_ego = state
    return 0.0

CONTROLLERS = {
  'open': OpenController,
  'simple': SimpleController,
  'c1': Controller,
}
