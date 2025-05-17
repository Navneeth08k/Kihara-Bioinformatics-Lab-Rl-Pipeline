import os
from amyloid_env_v0 import AmyloidEnv, load_structure  # assuming your code is in amyloid_env.py
import numpy as np

def test_environment(sample_dir="data"):
    from amyloid_env_v0 import get_all_samples
    pairs = get_all_samples(sample_dir)
    assert len(pairs) > 0, "No CIF sample pairs found."

    for i, (af2, exp) in enumerate(pairs):
        print(f"\n--- Testing Sample {i+1} ---")
        env = AmyloidEnv(af2, exp)
        
        obs = env.reset()
        assert isinstance(obs, np.ndarray), "Reset did not return a NumPy array"
        assert obs.shape[0] % 2 == 0, "Angle vector should be even-sized (phi/psi pairs)"
        
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)

        assert next_obs.shape == obs.shape, "State shape changed unexpectedly"
        assert isinstance(reward, float), "Reward is not a float"
        assert isinstance(done, bool), "Done flag is not boolean"

        print("Reset, step, and reward passed.")
        env.render()

if __name__ == "__main__":
    test_environment()
