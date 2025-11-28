import unittest
import torch
import numpy as np
from src.cublino_contra.model import PolicyValueNet
from src.cublino_contra.mcts import MCTS
from src.cublino_contra.env import CublinoContraEnv

class TestCublinoAI(unittest.TestCase):
    def test_model_shapes(self):
        """Test if the model accepts the correct input and produces correct output shapes."""
        batch_size = 2
        board_size = 7
        model = PolicyValueNet(board_size=board_size)
        
        # Input: (Batch, 3, 7, 7)
        dummy_input = torch.randn(batch_size, 3, board_size, board_size)
        
        policy, value = model(dummy_input)
        
        # Policy shape: (Batch, 196)
        self.assertEqual(policy.shape, (batch_size, 196))
        
        # Value shape: (Batch, 1)
        self.assertEqual(value.shape, (batch_size, 1))

    def test_mcts_integration(self):
        """Test if MCTS can run a few simulations with the model."""
        env = CublinoContraEnv()
        model = PolicyValueNet()
        model.eval()

        def policy_value_fn(state):
            # State is the env
            board = state.board
            # Convert to tensor: (1, 3, 7, 7)
            # Board is (7, 7, 3), need to transpose to (3, 7, 7)
            board_tensor = torch.FloatTensor(board).permute(2, 0, 1).unsqueeze(0)
            
            with torch.no_grad():
                log_act_probs, value = model(board_tensor)
                
            act_probs = np.exp(log_act_probs.numpy().flatten())
            value = value.item()
            
            # Create list of (action, prob)
            # For now, return all actions. In real usage, we might mask invalid ones.
            # But MCTS handles invalid moves by trying them and getting a loss/penalty or just failing?
            # Wait, my MCTS implementation calls `state.step(action)`.
            # If action is invalid, `env.step` returns info['error'].
            # But `env.step` in CublinoContraEnv returns `terminated=True` and penalty for invalid moves?
            # Let's check env.py.
            # Yes: return self._get_obs(), -10, True, False, {"error": ...}
            # So invalid moves result in immediate termination with -10 reward.
            # MCTS will learn to avoid them.
            
            return zip(range(len(act_probs)), act_probs), value

        mcts = MCTS(policy_value_fn, n_playout=10) # Low playout for test
        
        # Run get_move_probs
        acts, probs = mcts.get_move_probs(env)
        
        self.assertEqual(len(acts), 196) # Should return probs for all actions (since we didn't filter)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
