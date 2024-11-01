import numpy as np
import importlib
from tinyphysics import TinyPhysicsSimulator
from typing import Dict, List
from tqdm.contrib.concurrent import process_map
from functools import partial


def run_single_eval(data_path: str, controller, steps: int, seed: int) -> Dict[str, float]:
    """Run a single evaluation with a specific seed"""
    np.random.seed(seed)
    controller = controller()
    sim = TinyPhysicsSimulator(str(data_path), controller=controller, debug=False)
    if getattr(controller, "giveSim", None):
      controller.giveSim(sim)
    np.random.seed(seed)
    return sim.rollout(steps)

def parallel_eval(path: str, controller, n_steps: int = 150, n_evals: int = 16) -> List[Dict[str, float]]:
    """
    Run parallel evaluations of a controller on a route
    
    Args:
        path: Path to the route data
        controller_type: Name of the controller to test
        n_steps: Number of steps to simulate
        n_evals: Number of parallel evaluations
    
    Returns:
        List of dictionaries containing costs for each evaluation
    """
    # Create different seeds for each evaluation
    base_seed = np.random.randint(0, 2**32)
    seeds = [(base_seed + i) % 2**32 for i in range(n_evals)]
    
    # Create partial function with fixed arguments
    eval_func = partial(run_single_eval, 
                       path,
                       controller,
                       n_steps)
    
    # Run evaluations in parallel
    results = process_map(eval_func, seeds, max_workers=16, chunksize=1)
    
    return results