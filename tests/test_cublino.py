import unittest
import numpy as np
from src.cublino_contra.env import CublinoContraEnv

class TestCublinoContra(unittest.TestCase):
    def setUp(self):
        self.env = CublinoContraEnv()
        self.env.reset()

    def test_setup(self):
        # Check P1 setup (Row 0)
        for c in range(7):
            p, top, south = self.env.board[0, c]
            self.assertEqual(p, 1)
            self.assertEqual(top, 6)
            self.assertEqual(south, 3)
        
        # Check P2 setup (Row 6)
        for c in range(7):
            p, top, south = self.env.board[6, c]
            self.assertEqual(p, -1)
            self.assertEqual(top, 6)
            self.assertEqual(south, 4)

    def test_movement_rotation(self):
        # P1 at (0, 0): Top=6, South=3.
        # Move North (Direction 0).
        # Action = (0*7 + 0)*4 + 0 = 0.
        obs, reward, done, truncated, info = self.env.step(0)
        
        # Should be at (1, 0)
        self.assertEqual(self.env.board[0, 0, 0], 0)
        p, top, south = self.env.board[1, 0]
        self.assertEqual(p, 1)
        
        # Rotation: North
        # Old Top=6, Old South=3. Old Bottom=1.
        # New Top = Old South = 3.
        # New South = Old Bottom = 1.
        self.assertEqual(top, 3)
        self.assertEqual(south, 1)

    def test_battle_mechanic(self):
        # Setup a battle scenario
        # Clear board
        self.env.board.fill(0)
        
        # Place P2 die at (3, 3). Top=1, South=2.
        self.env.board[3, 3] = [-1, 1, 2]
        
        # Place P1 die at (2, 3). Top=6, South=3. (Neighbor South)
        self.env.board[2, 3] = [1, 6, 3]
        
        # Place P1 die at (3, 2). Top=5, South=1. (Neighbor West)
        # We need to move a P1 die into (3, 2) to trigger battle?
        # Or just have them there?
        # The battle triggers when a move is made.
        # Let's say P1 moves a die from (3, 1) to (3, 2).
        self.env.board[3, 1] = [1, 4, 1] # Top=4, South=1. East=2?
        # Let's check East face for (4, 1).
        # Top=4 (-X), South=1 (+Z). Cross: (-1,0,0)x(0,0,1) = (0,1,0) = 5 (North).
        # So East=5.
        # Move East (Dir 1).
        # New Top = Old West = 7-East = 2.
        # New South = 1.
        
        # Action: Move (3, 1) East.
        # Index = (3*7 + 1)*4 + 1 = (22)*4 + 1 = 89.
        
        # Before move:
        # P2 at (3, 3) (Defender). Value=1.
        # P1 at (2, 3). Value=6.
        # P1 moving to (3, 2). New Value=2.
        
        # Battle Calculation:
        # Defender (P2) at (3, 3).
        # Neighbors: (2, 3) [P1, Val=6], (3, 2) [P1, Val=2].
        # Friendly Neighbors: None.
        # Defender Total = 1 + 0 = 1.
        # Attacker Total = 6 + 2 = 8.
        # 1 < 8 -> Defender should be removed.
        
        self.env.current_player = 1
        obs, reward, done, truncated, info = self.env.step(89)
        
        # Check P2 removed
        self.assertEqual(self.env.board[3, 3, 0], 0)
        # Check P1 moved
        self.assertEqual(self.env.board[3, 2, 0], 1)
        self.assertEqual(self.env.board[3, 2, 1], 2)

    def test_battle_survival(self):
        # Setup a battle scenario where Defender survives
        self.env.board.fill(0)
        
        # P2 at (3, 3). Top=6.
        self.env.board[3, 3] = [-1, 6, 2]
        
        # P1 at (2, 3). Top=1.
        self.env.board[2, 3] = [1, 1, 3]
        
        # P1 moves to (3, 2). Top=1.
        self.env.board[3, 1] = [1, 5, 1] # Move East -> Top=2?
        # Top=5, South=1. East=3?
        # Top=5 (+Y), South=1 (+Z). Cross: (0,1,0)x(0,0,1) = (1,0,0) = 3 (East).
        # Move East -> New Top = Old West = 4.
        
        # Action: Move (3, 1) East.
        
        # Defender Total = 6.
        # Attacker Total = 1 + 4 = 5.
        # 6 >= 5 -> Survival.
        
        self.env.current_player = 1
        action = (3*7 + 1)*4 + 1
        self.env.step(action)
        
        # Check P2 still there
        self.assertEqual(self.env.board[3, 3, 0], -1)

    def test_win_condition(self):
        # P1 at (5, 0). Move North to (6, 0).
        self.env.board.fill(0)
        self.env.board[5, 0] = [1, 6, 3] # Top=6, South=3
        
        # Action: Move (5, 0) North.
        # Index = (5*7 + 0)*4 + 0 = 35*4 = 140.
        
        obs, reward, done, truncated, info = self.env.step(140)
        
        self.assertTrue(done)
        self.assertEqual(reward, 1)
        self.assertEqual(info['winner'], 1)

if __name__ == '__main__':
    unittest.main()
