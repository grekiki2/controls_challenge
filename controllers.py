from collections import namedtuple
from common import *

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

class Controller2(BaseController):
  def __init__(self):
    self.state_history = []
    self.action_history = []
    self.lat_accel_history = []
    self.model = TinyPhysicsModel("./models/tinyphysics.onnx")
  def update(self, target_lataccel, current_lataccel, state, active, last_action):
    # roll_lataccel, v_ego, a_ego = state
    if active:
      action = self.determineBestAction(state, current_lataccel, target_lataccel)
    else:
      action = last_action
    self.state_history.append(state)
    self.action_history.append(action)
    self.lat_accel_history.append(current_lataccel)
    return action

  def determineBestAction(self, state, cur_lat_accel, target_lataccel):
    return 0

CONTROLLERS = {
  'open': OpenController,
  'simple': SimpleController,
  'c1': Controller,
  'c2': Controller2,
}
