import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from hashlib import md5
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

from common import *
from controllers import BaseController, CONTROLLERS
import controllers
import signal

sns.set_theme()
signal.signal(signal.SIGINT, signal.SIG_DFL)  # Enable Ctrl-C on plot windows


class TinyPhysicsSimulator:
  def __init__(self, data_path: str, controller: BaseController, debug: bool = False, rng_seed:bool=True) -> None:
    self.data_path = data_path
    self.data = self.get_data(data_path)
    self.controller = controller
    self.debug = debug
    self.times = []
    self.reset(rng_seed)
    controllers.sim = self

  def reset(self, rng_seed:bool=False) -> None:
    self.step_idx = CONTEXT_LENGTH
    self.state_history = [self.get_state_target(i)[0] for i in range(self.step_idx)]
    self.action_history:List[float] = self.data['steer_command'].values[:self.step_idx].tolist()
    self.current_lataccel_history = [self.get_state_target(i)[1] for i in range(self.step_idx)]
    self.target_lataccel_history = [self.get_state_target(i)[1] for i in range(self.step_idx)]
    self.current_lataccel = self.current_lataccel_history[-1]
    if rng_seed:
      seed = int(md5(self.data_path.encode()).hexdigest(), 16) % 10**4
      np.random.seed(seed)

  def get_data(self, data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    processed_df = pd.DataFrame({
      'roll_lataccel': np.sin(df['roll'].values) * ACC_G,
      'v_ego': df['vEgo'].values,
      'a_ego': df['aEgo'].values,
      'target_lataccel': df['targetLateralAcceleration'].values,
      'steer_command': df['steerCommand'].values
    })
    return processed_df

  def control_step(self, step_idx: int) -> None:
    if step_idx >= CONTROL_START_IDX:
      action = self.controller.update(self.target_lataccel_history[step_idx], self.current_lataccel, self.state_history[step_idx], True, self.action_history[step_idx-1])
    else:
      # self.controller.update(self.target_lataccel_history[step_idx], self.current_lataccel, self.state_history[step_idx], False, self.action_history[step_idx-1])
      action = self.data['steer_command'].values[step_idx]
    action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
    self.action_history.append(action)

  def sim_step(self, step_idx: int) -> None:
    pred_lataccel = model.get_current_lataccel(
      sim_states=self.state_history[-CONTEXT_LENGTH:],
      actions=self.action_history[-CONTEXT_LENGTH:],
      past_preds=self.current_lataccel_history[-CONTEXT_LENGTH:]
    )
    if step_idx >= CONTROL_START_IDX:
      self.current_lataccel = np.clip(pred_lataccel, self.current_lataccel - MAX_ACC_DELTA, self.current_lataccel + MAX_ACC_DELTA)
    else:
      self.current_lataccel = self.get_state_target(step_idx)[1]

    self.current_lataccel_history.append(self.current_lataccel)

  def get_state_target(self, step_idx: int) -> Tuple[State, float]:
    state = self.data.iloc[step_idx]
    return State(roll_lataccel=state['roll_lataccel'], v_ego=state['v_ego'], a_ego=state['a_ego']), state['target_lataccel']

  def step(self) -> None:
    state, target = self.get_state_target(self.step_idx)
    self.state_history.append(state)
    self.target_lataccel_history.append(target)
    self.control_step(self.step_idx)
    self.sim_step(self.step_idx)
    self.step_idx += 1

  def plot_data(self, ax, lines, axis_labels, title) -> None:
    ax.clear()
    for line, label in lines:
      ax.plot(line, label=label)
    ax.axline((CONTROL_START_IDX, 0), (CONTROL_START_IDX, 1), color='black', linestyle='--', alpha=0.5, label='Control Start')
    ax.legend()
    ax.set_title(f"{title} | Step: {self.step_idx}")
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])

  def compute_cost(self) -> float:
    target = np.array(self.target_lataccel_history)[CONTROL_START_IDX:]
    pred = np.array(self.current_lataccel_history)[CONTROL_START_IDX:]

    lat_accel_cost = np.mean((target - pred)**2) * 100
    jerk_cost = np.mean((np.diff(pred) / DEL_T)**2) * 100
    total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
    return {'lataccel_cost': lat_accel_cost, 'jerk_cost': jerk_cost, 'total_cost': total_cost}

  def rollout(self) -> None:
    if self.debug:
      fig, ax = plt.subplots(4, figsize=(12, 14), constrained_layout=True)

    for _ in range(CONTEXT_LENGTH, len(self.data)-CONTEXT_LENGTH):
      self.step()
      if self.debug and self.step_idx % 10 == 0:
        print(f"Step {self.step_idx:<5}: Current lataccel: {self.current_lataccel:>6.2f}, Target lataccel: {self.target_lataccel_history[-1]:>6.2f}")
        self.plot_data(ax[0], [(self.target_lataccel_history, 'Target lataccel'), (self.current_lataccel_history, 'Current lataccel')], ['Step', 'Lateral Acceleration'], 'Lateral Acceleration')
        self.plot_data(ax[1], [(self.action_history, 'Action')], ['Step', 'Action'], 'Action')
        self.plot_data(ax[2], [(np.array(self.state_history)[:, 0], 'Roll Lateral Acceleration')], ['Step', 'Lateral Accel due to Road Roll'], 'Lateral Accel due to Road Roll')
        self.plot_data(ax[3], [(np.array(self.state_history)[:, 1], 'v_ego')], ['Step', 'v_ego'], 'v_ego')
        plt.pause(0.01)

    # if self.debug:
    #   plt.show()
    cost = self.compute_cost()
    print(cost)
    return cost


model = TinyPhysicsModel("./models/tinyphysics.onnx")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_path", type=str, default="./data")
  parser.add_argument("--num_segs", type=int, default=100)
  parser.add_argument("--debug", action='store_true')
  parser.add_argument("--controller", default='simple', choices=CONTROLLERS.keys())
  args = parser.parse_args()

  controller = CONTROLLERS[args.controller]()

  data_path = Path(args.data_path)
  if data_path.is_file():
    sim = TinyPhysicsSimulator(args.data_path, controller=controller, debug=args.debug)
    costs = sim.rollout()
    print(f"\nAverage lataccel_cost: {costs['lataccel_cost']:>6.4}, average jerk_cost: {costs['jerk_cost']:>6.4}, average total_cost: {costs['total_cost']:>6.4}")
  elif data_path.is_dir():
    costs = []
    files = sorted(data_path.iterdir())[:args.num_segs]
    for data_file in tqdm(files, total=len(files)):
      sim = TinyPhysicsSimulator(str(data_file), controller=controller, debug=args.debug)
      cost = sim.rollout()
      costs.append(cost)
    costs_df = pd.DataFrame(costs)
    print(f"\nAverage lataccel_cost: {np.mean(costs_df['lataccel_cost']):>6.4}, average jerk_cost: {np.mean(costs_df['jerk_cost']):>6.4}, average total_cost: {np.mean(costs_df['total_cost']):>6.4}")
    for cost in costs_df.columns:
      plt.hist(costs_df[cost], bins=np.arange(0, 1000, 10), label=cost, alpha=0.5)
    plt.xlabel('costs')
    plt.ylabel('Frequency')
    plt.title('costs Distribution')
    plt.legend()
    plt.show()
