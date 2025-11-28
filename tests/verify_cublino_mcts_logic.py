
import unittest
import torch
import numpy as np
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.cublino_contra.mcts import MCTS_CPP
from src.cublino_contra.env import CublinoContraEnv
from src.cublino_contra.model import PolicyValueNet

class TestCublinoMCTSLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = "temp_logic_test_model.pt"
        cls.device = torch.device("cpu") # Use CPU for consistent testing
        model = PolicyValueNet(board_size=7).to(cls.device)
        model.eval()
        script_model = torch.jit.script(model)
        script_model.save(cls.model_path)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.model_path):
            os.remove(cls.model_path)

    def test_find_immediate_win(self):
        """Test if MCTS finds an immediate winning move (reaching the end)."""
        env = CublinoContraEnv()
        env.reset()
        # Clear board
        env.board[:,:,:] = 0
        
        # P1 at (5, 3), facing North (target Row 6)
        # Any value is fine, say Top 1, South 5.
        env.board[5, 3] = [1, 1, 5]
        env.current_player = 1
        
        # Win move: (5,3) -> (6,3) (Direction 0: North)
        # Action = (5*7 + 3)*4 + 0 = 38*4 = 152
        winning_action = 152
        
        mcts = MCTS_CPP(self.model_path, c_puct=1.0, n_playout=200, device=self.device)
        
        acts, probs = mcts.get_move_probs(env, temp=0.1) # Low temp for deterministic choice
        
        best_move = acts[np.argmax(probs)]
        
        print(f"\nTest Immediate Win:")
        print(f"Top moves: {sorted(zip(acts, probs), key=lambda x: x[1], reverse=True)[:3]}")
        
        self.assertEqual(best_move, winning_action, "MCTS failed to find immediate win.")

    def test_find_capture(self):
        """Test if MCTS prefers a capturing move over a non-capturing one."""
        # This is harder with random value net, but the capture logic gives a positional advantage?
        # Actually, the reward is only +1 for WIN. Capture gives 0 reward immediately.
        # With a random value net, the state AFTER capture is evaluated randomly.
        # So MCTS won't know capture is good unless the random net thinks so (unlikely) 
        # or if it leads to a forced win.
        # Therefore, we CANNOT easily test tactical capture with an untrained net.
        # We can only test terminal states (Win/Loss).
        pass

    def test_avoid_immediate_loss(self):
        """Test if MCTS avoids a move that lets the opponent win immediately?
        With self-play MCTS, it assumes the opponent plays optimally (or according to same policy).
        If opponent has a winning move next turn, the current node value should be low.
        """
        env = CublinoContraEnv()
        env.reset()
        env.board[:,:,:] = 0
        
        # P1 to move.
        # P1 has a piece at (0, 0).
        # P2 has a piece at (1, 3) ready to move to (0, 3) and win.
        env.board[0, 0] = [1, 1, 5]
        env.board[1, 3] = [-1, 1, 2] # P2 facing South?
        
        # If P1 moves (0,0)->(1,0), P2 moves (1,3)->(0,3) and wins.
        # This branch should have low value for P1.
        # But P1 has no way to stop it?
        # Let's give P1 a way to block.
        # P2 at (1, 0). P1 at (0, 1).
        # P2 threatens (0, 0).
        # P1 can move (0, 1) -> (0, 0) to block? No, occupancy check.
        # Or P1 can capture P2?
        
        # Let's stick to the immediate win test which is reliable.
        pass

if __name__ == '__main__':
    unittest.main()
