import torch
import numpy as np
from tic_tac_toe_bolt.model import PolicyValueNet

def test_model_forward():
    model = PolicyValueNet()
    
    # Create a dummy input (batch_size=1, channels=3, height=3, width=3)
    # Channel 0: Player 1 marks
    # Channel 1: Player 2 marks
    # Channel 2: Turn (all 1s)
    dummy_input = torch.zeros((1, 3, 3, 3))
    
    # Forward pass
    policy, value = model(dummy_input)
    
    print("Policy shape:", policy.shape)
    print("Value shape:", value.shape)
    print("Policy output:", policy)
    print("Value output:", value)
    
    assert policy.shape == (1, 9), "Policy shape mismatch"
    assert value.shape == (1, 1), "Value shape mismatch"
    
    print("Model forward pass verified successfully!")

if __name__ == "__main__":
    test_model_forward()
