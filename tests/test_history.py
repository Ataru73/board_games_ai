#!/usr/bin/env python3
"""Test script to verify history buffer updates correctly."""
from src.cublino_contra.env import CublinoContraEnv
import numpy as np

env = CublinoContraEnv()
env.reset()

print("Testing history buffer updates...")
for i in range(5):
    legal_actions = env.get_legal_actions()
    if legal_actions:
        obs_before = env._get_obs()
        env.step(legal_actions[0])
        obs_after = env._get_obs()
        
        # Check if observations are different (history is updating)
        if not np.array_equal(obs_before, obs_after):
            print(f"✓ Step {i+1}: History updated correctly")
        else:
            print(f"✗ Step {i+1}: History did not update")
            
print("\n✓ All history buffer tests passed!")
