import onnxruntime as ort
from collections import namedtuple
from typing import List, Union, Tuple
import numpy as np

ACC_G = 9.81
CONTROL_START_IDX = 100
CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
LAT_ACCEL_COST_MULTIPLIER = 5.0

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])


class LataccelTokenizer:
  def __init__(self):
    self.vocab_size = VOCAB_SIZE
    self.bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], self.vocab_size)

  def encode(self, value: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
    value = self.clip(value)
    return np.digitize(value, self.bins, right=True)

  def decode(self, token: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    return self.bins[token]

  def clip(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.clip(value, LATACCEL_RANGE[0], LATACCEL_RANGE[1])


class TinyPhysicsModel:
  def __init__(self, model_path: str) -> None:
    self.tokenizer = LataccelTokenizer()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3

    with open(model_path, "rb") as f:
      self.ort_session = ort.InferenceSession(f.read(), options, ['CPUExecutionProvider'])

  def softmax(self, x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

  def predict(self, input_data: dict, temperature=1.) -> dict:
    res = self.ort_session.run(None, input_data)[0]
    probs = self.softmax(res / temperature, axis=-1)
    # we only care about the last timestep (batch size is just 1)
    assert probs.shape[0] == 1
    assert probs.shape[2] == VOCAB_SIZE
    lat_accel_pred = np.random.choice(probs.shape[2], p=probs[0, -1])
    return lat_accel_pred

  def input_data(self, sim_states: List[State], actions: List[float], past_preds: List[float]):
    tokenized_past_preds = self.tokenizer.encode(past_preds)
    raw_states = [list(x) for x in sim_states]
    states = np.column_stack([actions, raw_states]) # [(action, roll_lataccel, v_ego, a_ego)]
    return {
      'states': np.expand_dims(states, axis=0).astype(np.float32),
      'tokens': np.expand_dims(tokenized_past_preds, axis=0).astype(np.int64)
    }
  # Inputs are cropped to CONTEXT_LENGTH. The only important function
  def get_current_lataccel(self, sim_states: List[State], actions: List[float], past_preds: List[float]) -> float:
    return self.tokenizer.decode(self.predict(self.input_data(sim_states, actions, past_preds)))
  
  def get_lataccel_logits(self, sim_states: List[State], actions: List[float], past_preds: List[float]) -> np.ndarray:
    return self.ort_session.run(None, self.input_data(sim_states, actions, past_preds))[0][0, -1]