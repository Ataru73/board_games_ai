import torch
import numpy as np

from cublino_contra import _mcts_cpp

def test_mcts_cpp_initialization():
    # Create a dummy model file (scripted module)
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            # Input: (B, 12, 7, 7)
            # Output: Policy (B, 196), Value (B, 1)
            B = x.shape[0]
            policy = torch.ones(B, 196) / 196
            value = torch.zeros(B, 1)
            return policy, value

    model = DummyModel()
    scripted_model = torch.jit.script(model)
    scripted_model.save("dummy_model.pt")

    # Initialize MCTS
    mcts = _mcts_cpp.MCTS("dummy_model.pt", 1.0, 10, "cpu")
    assert mcts is not None

def test_state_history_and_obs():
    # Create a dummy model
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.ones(x.shape[0], 196), torch.zeros(x.shape[0], 1)
    
    scripted_model = torch.jit.script(DummyModel())
    scripted_model.save("dummy_model.pt")
    
    mcts = _mcts_cpp.MCTS("dummy_model.pt", 1.0, 10, "cpu")
    
    # Create state
    state = _mcts_cpp.CublinoState()
    
    # Check initial observation shape
    obs = state.get_obs()
    assert obs.shape == (1, 12, 7, 7)
    
    # Verify channels 0-2 match channels 9-11 (initially all same)
    assert torch.allclose(obs[0, 0:3], obs[0, 9:12])
    
    # Make a move
    # P1 starts. Valid move: (0,0) -> South (2) is invalid for P1? No, P1 is at row 0.
    # P1 pieces at row 0. Move South (dir 2) -> Row -1 (Invalid).
    # Move North (dir 0) -> Row 1. Valid.
    # Action = (0 * 7 + 0) * 4 + 0 = 0.
    
    # Legal actions check
    legal = state.get_legal_actions()
    assert len(legal) > 0
    
    action = legal[0]
    state.step(action)
    
    # Check observation again
    obs_new = state.get_obs()
    assert obs_new.shape == (1, 12, 7, 7)
    
    # Now history should be different
    # Channel 9-11 (newest) should differ from 6-8 (previous)
    # Actually, 6-8 should be the state BEFORE the move.
    # 9-11 is state AFTER the move.
    assert not torch.allclose(obs_new[0, 6:9], obs_new[0, 9:12])
    
    # 0-2 should still be initial state
    # 3-5 should be initial state
    # 6-8 should be initial state (since we pushed initial state 4 times, then popped one)
    # Wait:
    # Init: [S0, S0, S0, S0]
    # Step 1: [S0, S0, S0, S1]
    # So 0-2 (S0), 3-5 (S0), 6-8 (S0), 9-11 (S1).
    assert torch.allclose(obs_new[0, 0:3], obs_new[0, 3:6])
    assert torch.allclose(obs_new[0, 3:6], obs_new[0, 6:9])

def test_set_state_from_python():
    state = _mcts_cpp.CublinoState()
    
    # Create a dummy 12-channel observation
    # 4 states. Let's make them distinct.
    obs_np = np.zeros((7, 7, 12), dtype=np.int32)
    obs_np[0, 0, 0] = 1 # State 0, Pos(0,0), Ch 0
    obs_np[0, 0, 3] = 2 # State 1, Pos(0,0), Ch 0 (offset 3)
    obs_np[0, 0, 6] = 3 # State 2
    obs_np[0, 0, 9] = 4 # State 3
    
    state.set_state_from_python(obs_np, 1)
    
    obs = state.get_obs()
    
    # Verify reconstruction
    assert obs[0, 0, 0, 0].item() == 1
    assert obs[0, 3, 0, 0].item() == 2
    assert obs[0, 6, 0, 0].item() == 3
    assert obs[0, 9, 0, 0].item() == 4

if __name__ == "__main__":
    # Manual run if pytest not available
    test_mcts_cpp_initialization()
    test_state_history_and_obs()
    test_set_state_from_python()
    print("All tests passed!")
