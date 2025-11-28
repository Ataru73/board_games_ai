
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

class TestCublinoDraw(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = "temp_draw_test_model.pt"
        cls.device = torch.device("cpu")
        model = PolicyValueNet(board_size=7).to(cls.device)
        model.eval()
        script_model = torch.jit.script(model)
        script_model.save(cls.model_path)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.model_path):
            os.remove(cls.model_path)

    def test_three_fold_repetition(self):
        """Test 3-fold repetition detection."""
        env = CublinoContraEnv()
        env.reset()
        env.board[:,:,:] = 0
        
        # Setup simple oscilation
        # P1 at (0,0), P2 at (6,6)
        # P1 moves (0,0) <-> (0,1)
        # P2 moves (6,6) <-> (6,5)
        
        # Initial State: 
        # P1(0,0): Top 1, S 5. 
        # P2(6,6): Top 1, S 2.
        
        env.board[0, 0] = [1, 1, 5]
        env.board[6, 6] = [-1, 1, 2]
        env.current_player = 1
        
        # Re-record initial state in history
        env.history.clear()
        env.history[env._get_state_key()] = 1
        
        # State 1 (Start) - Count: 1
        
        # Move 1: P1 (0,0)->(0,1) (East)
        # Action: (0*7+0)*4 + 1 = 1
        _, _, term, _, _ = env.step(1)
        self.assertFalse(term)
        
        # Move 2: P2 (6,6)->(6,5) (West)
        # Action: (6*7+6)*4 + 3 = 195
        _, _, term, _, _ = env.step(195)
        self.assertFalse(term)
        
        # Move 3: P1 (0,1)->(0,0) (West)
        # Action: (0*7+1)*4 + 3 = 7
        _, _, term, _, _ = env.step(7) # Back to P1 start pos? 
        # Note: Dice rotate. Does returning to square return to same orientation?
        # (0,0) Top 1, S 5. -> East -> (0,1). Top ?, S 5.
        # -> West -> (0,0). Top ?, S 5.
        # East: New Top = Old West.
        # West: New Top = Old East.
        # 1, 5. East -> Top=West. West of (1,5)?
        # 1(Z+), 5(Y+). E(X+)=3. W(X-)=4.
        # So after East: Top=4.
        # At (0,1): Top=4, S=5.
        # West: New Top = Old East.
        # 4(X-), 5(Y+). East=1(Z+).
        # So after West: Top=1.
        # Yes! For this specific move sequence (East then West), the die returns to original state.
        
        self.assertFalse(term)
        
        # Move 4: P2 (6,5)->(6,6) (East)
        # Action: (6*7+5)*4 + 1 = 189
        _, _, term, _, _ = env.step(189)
        self.assertFalse(term)
        
        # State 1 Repeated - Count: 2
        
        # Move 5: P1 (0,0)->(0,1)
        env.step(1)
        # Move 6: P2 (6,6)->(6,5)
        env.step(195)
        # Move 7: P1 (0,1)->(0,0)
        env.step(7)
        # Move 8: P2 (6,5)->(6,6)
        obs, reward, terminated, truncated, info = env.step(189)
        
        # State 1 Repeated - Count: 3!
        
        print(f"\nFinal Info: {info}")
        self.assertTrue(terminated, "Game should terminate on 3rd repetition.")
        self.assertEqual(reward, 0, "Reward should be 0 for draw.")
        self.assertEqual(info.get('winner'), 0, "Winner should be 0.")

if __name__ == '__main__':
    unittest.main()
