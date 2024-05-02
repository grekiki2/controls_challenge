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

DELTA = 10/1024
class Controller2(BaseController):
  def __init__(self):
    self.state_history = []
    self.action_history = []
    self.lat_accel_history = []
    self.model = TinyPhysicsModel("./models/tinyphysics.onnx")
    self.last_action = 0
    self.running = False

  def update(self, target_lataccel, current_lataccel, state, active, last_action):
    self.logs = {}
    # roll_lataccel, v_ego, a_ego = state
    self.state_history.append(state)
    self.lat_accel_history.append(current_lataccel)
    if active:
      if not self.running:
        self.action_history.append(last_action)
        self.running = True
      action = self.determineBestAction(target_lataccel)
    else:
      action = last_action
    self.action_history.append(action)
    return action

  def getActionCost(self, target_lataccel, action):
    logits = self.model.get_lataccel_logits(self.state_history[-20:], self.action_history[-19:]+[action], self.lat_accel_history[-20:])
    probs = self.model.softmax(logits, axis=-1)
    # expected_cost = 0
    # for i in range(1024):
    #   accel = self.model.tokenizer.decode(i)
    #   # expected_cost += (accel - target_lataccel)**2 * probs[i] * 100 * 5
    #   expected_cost += (accel - 0)**2 * probs[i] * 100 * 100
    # return expected_cost
    return (action-target_lataccel)**2

  def determineBestAction(self, target_lataccel):
    cost = 10000000000
    best_action = None
    last_act = self.action_history[-1]
    xs = []
    ys = []
    for action in np.linspace(-0.5, 0.5, 1000):
      expected_cost = self.getActionCost(target_lataccel, action)
      xs.append(action)
      ys.append(expected_cost)
      if expected_cost < cost:
        cost = expected_cost
        best_action = action
    self.logs["x"] = xs
    self.logs["y"] = ys
    print(f"Best action: {best_action}, cost: {cost}")
    self.model.get_lataccel_logits(self.state_history[-20:], self.action_history[-19:]+[best_action], self.lat_accel_history[-20:], True)
    return best_action

CONTROLLERS = {
  'open': OpenController,
  'simple': SimpleController,
  'c1': Controller,
  'c2': Controller2,
}
