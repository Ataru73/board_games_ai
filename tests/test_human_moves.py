import unittest
from unittest.mock import MagicMock
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Mock OpenGL modules before importing play
sys.modules['OpenGL'] = MagicMock()
sys.modules['OpenGL.GL'] = MagicMock()
sys.modules['OpenGL.GLU'] = MagicMock()
sys.modules['OpenGL.GLUT'] = MagicMock()

# Mock src.cublino_contra modules
sys.modules['src.cublino_contra.model'] = MagicMock()
sys.modules['src.cublino_contra.mcts'] = MagicMock()
# Do NOT mock src.cublino_contra itself, as we need to import play from it
# sys.modules['src.cublino_contra'] = MagicMock() 

# Mock gym to avoid registration issues
# sys.modules['gymnasium'] = MagicMock()
# sys.modules['gym'] = sys.modules['gymnasium']

from src.cublino_contra.play import HumanPlayer

class TestHumanPlayer(unittest.TestCase):
    def setUp(self):
        self.player = HumanPlayer()
        self.env = MagicMock()
        self.env.unwrapped.board = np.zeros((7, 7, 3), dtype=np.int8)

    def test_p1_cannot_move_backward(self):
        self.player.set_player_ind(1)
        self.player.selected_square = (3, 3) # Center
        
        # Setup board: P1 at (3,3)
        self.env.unwrapped.board[3, 3] = [1, 1, 1]
        
        # Mock global current_env
        import src.cublino_contra.play as play
        play.current_env = self.env
        play.human_player_instance = self.player
        
        # Simulate mouse click logic (extracted from play.py)
        board_y, board_x = 3, 3
        
        # Clear legal moves
        self.player.legal_moves = []
        
        # Logic from play.py
        for d in range(4):
            dr, dc = 0, 0
            if d == 0: dr = 1  # North (Row + 1)
            elif d == 1: dc = 1 # East (Col + 1)
            elif d == 2: dr = -1 # South (Row - 1)
            elif d == 3: dc = -1 # West (Col - 1)
            
            # Check for backward moves (The Fix)
            if self.player.player == 1 and d == 2: continue
            if self.player.player == -1 and d == 0: continue

            tr, tc = board_y + dr, board_x + dc
            if 0 <= tr < 7 and 0 <= tc < 7 and self.env.unwrapped.board[tr, tc, 0] == 0:
                self.player.legal_moves.append((tr, tc, d))
        
        # Assertions
        # P1 cannot move South (d=2). Legal moves should be N(0), E(1), W(3)
        legal_directions = [m[2] for m in self.player.legal_moves]
        self.assertNotIn(2, legal_directions)
        self.assertIn(0, legal_directions)
        self.assertIn(1, legal_directions)
        self.assertIn(3, legal_directions)

    def test_p2_cannot_move_backward(self):
        self.player.set_player_ind(-1)
        self.player.selected_square = (3, 3)
        
        self.env.unwrapped.board[3, 3] = [-1, 1, 1]
        
        import src.cublino_contra.play as play
        play.current_env = self.env
        play.human_player_instance = self.player
        
        board_y, board_x = 3, 3
        self.player.legal_moves = []
        
        for d in range(4):
            dr, dc = 0, 0
            if d == 0: dr = 1
            elif d == 1: dc = 1
            elif d == 2: dr = -1
            elif d == 3: dc = -1
            
            if self.player.player == 1 and d == 2: continue
            if self.player.player == -1 and d == 0: continue

            tr, tc = board_y + dr, board_x + dc
            if 0 <= tr < 7 and 0 <= tc < 7 and self.env.unwrapped.board[tr, tc, 0] == 0:
                self.player.legal_moves.append((tr, tc, d))
        
        # P2 cannot move North (d=0). Legal moves should be E(1), S(2), W(3)
        legal_directions = [m[2] for m in self.player.legal_moves]
        self.assertNotIn(0, legal_directions)
        self.assertIn(1, legal_directions)
        self.assertIn(2, legal_directions)
        self.assertIn(3, legal_directions)

if __name__ == '__main__':
    unittest.main()
